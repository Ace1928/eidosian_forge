from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_module_init_func(self, imported_modules, env, code):
    subfunction = self.mod_init_subfunction(self.pos, self.scope, code)
    self.generate_pymoduledef_struct(env, code)
    code.enter_cfunc_scope(self.scope)
    code.putln('')
    code.putln(UtilityCode.load_as_string('PyModInitFuncType', 'ModuleSetupCode.c')[0])
    if env.module_name.isascii():
        py2_mod_name = env.module_name
        fail_compilation_in_py2 = False
    else:
        fail_compilation_in_py2 = True
        py2_mod_name = env.module_name.encode('ascii', errors='ignore').decode('utf8')
    header2 = '__Pyx_PyMODINIT_FUNC init%s(void)' % py2_mod_name
    header3 = '__Pyx_PyMODINIT_FUNC %s(void)' % self.mod_init_func_cname('PyInit', env)
    header3 = EncodedString(header3)
    code.putln('#if PY_MAJOR_VERSION < 3')
    code.putln('%s CYTHON_SMALL_CODE; /*proto*/' % header2)
    if fail_compilation_in_py2:
        code.putln('#error "Unicode module names are not supported in Python 2";')
    if self.scope.is_package:
        code.putln('#if !defined(CYTHON_NO_PYINIT_EXPORT) && (defined(_WIN32) || defined(WIN32) || defined(MS_WINDOWS))')
        code.putln('__Pyx_PyMODINIT_FUNC init__init__(void) { init%s(); }' % py2_mod_name)
        code.putln('#endif')
    code.putln(header2)
    code.putln('#else')
    code.putln('%s CYTHON_SMALL_CODE; /*proto*/' % header3)
    if self.scope.is_package:
        code.putln('#if !defined(CYTHON_NO_PYINIT_EXPORT) && (defined(_WIN32) || defined(WIN32) || defined(MS_WINDOWS))')
        code.putln('__Pyx_PyMODINIT_FUNC PyInit___init__(void) { return %s(); }' % self.mod_init_func_cname('PyInit', env))
        code.putln('#endif')
    wrong_punycode_module_name = self.wrong_punycode_module_name(env.module_name)
    if wrong_punycode_module_name:
        code.putln('#if !defined(CYTHON_NO_PYINIT_EXPORT) && (defined(_WIN32) || defined(WIN32) || defined(MS_WINDOWS))')
        code.putln('void %s(void) {} /* workaround for https://bugs.python.org/issue39432 */' % wrong_punycode_module_name)
        code.putln('#endif')
    code.putln(header3)
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('{')
    code.putln('return PyModuleDef_Init(&%s);' % Naming.pymoduledef_cname)
    code.putln('}')
    mod_create_func = UtilityCode.load_as_string('ModuleCreationPEP489', 'ModuleSetupCode.c')[1]
    code.put(mod_create_func)
    code.putln('')
    code.putln('static CYTHON_SMALL_CODE int %s(PyObject *%s)' % (self.module_init_func_cname(), Naming.pymodinit_module_arg))
    code.putln('#endif')
    code.putln('#endif')
    code.putln('{')
    code.putln('int stringtab_initialized = 0;')
    code.putln('#if CYTHON_USE_MODULE_STATE')
    code.putln('int pystate_addmodule_run = 0;')
    code.putln('#endif')
    tempdecl_code = code.insertion_point()
    profile = code.globalstate.directives['profile']
    linetrace = code.globalstate.directives['linetrace']
    if profile or linetrace:
        if linetrace:
            code.use_fast_gil_utility_code()
        code.globalstate.use_utility_code(UtilityCode.load_cached('Profile', 'Profile.c'))
    code.put_declare_refcount_context()
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('if (%s) {' % Naming.module_cname)
    code.putln('if (%s == %s) return 0;' % (Naming.module_cname, Naming.pymodinit_module_arg))
    code.putln('PyErr_SetString(PyExc_RuntimeError, "Module \'%s\' has already been imported. Re-initialisation is not supported.");' % env.module_name.as_c_string_literal()[1:-1])
    code.putln('return -1;')
    code.putln('}')
    code.putln('#elif PY_MAJOR_VERSION >= 3')
    code.putln('if (%s) return __Pyx_NewRef(%s);' % (Naming.module_cname, Naming.module_cname))
    code.putln('#endif')
    code.putln('/*--- Module creation code ---*/')
    self.generate_module_creation_code(env, code)
    if profile or linetrace:
        tempdecl_code.put_trace_declarations()
        code.put_trace_frame_init()
    refnanny_import_code = UtilityCode.load_as_string('ImportRefnannyAPI', 'ModuleSetupCode.c')[1]
    code.putln(refnanny_import_code.rstrip())
    code.put_setup_refcount_context(header3)
    env.use_utility_code(UtilityCode.load('CheckBinaryVersion', 'ModuleSetupCode.c'))
    code.put_error_if_neg(self.pos, '__Pyx_check_binary_version(__PYX_LIMITED_VERSION_HEX, __Pyx_get_runtime_version(), CYTHON_COMPILING_IN_LIMITED_API)')
    code.putln('#ifdef __Pxy_PyFrame_Initialize_Offsets')
    code.putln('__Pxy_PyFrame_Initialize_Offsets();')
    code.putln('#endif')
    code.putln('%s = PyTuple_New(0); %s' % (Naming.empty_tuple, code.error_goto_if_null(Naming.empty_tuple, self.pos)))
    code.putln('%s = PyBytes_FromStringAndSize("", 0); %s' % (Naming.empty_bytes, code.error_goto_if_null(Naming.empty_bytes, self.pos)))
    code.putln('%s = PyUnicode_FromStringAndSize("", 0); %s' % (Naming.empty_unicode, code.error_goto_if_null(Naming.empty_unicode, self.pos)))
    for ext_type in ('CyFunction', 'FusedFunction', 'Coroutine', 'Generator', 'AsyncGen', 'StopAsyncIteration'):
        code.putln('#ifdef __Pyx_%s_USED' % ext_type)
        code.put_error_if_neg(self.pos, '__pyx_%s_init(%s)' % (ext_type, env.module_cname))
        code.putln('#endif')
    code.putln('/*--- Library function declarations ---*/')
    if env.directives['np_pythran']:
        code.put_error_if_neg(self.pos, '_import_array()')
    code.putln('/*--- Threads initialization code ---*/')
    code.putln('#if defined(WITH_THREAD) && PY_VERSION_HEX < 0x030700F0 && defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS')
    code.putln('PyEval_InitThreads();')
    code.putln('#endif')
    code.putln('/*--- Initialize various global constants etc. ---*/')
    code.put_error_if_neg(self.pos, '__Pyx_InitConstants()')
    code.putln('stringtab_initialized = 1;')
    code.put_error_if_neg(self.pos, '__Pyx_InitGlobals()')
    code.putln('#if PY_MAJOR_VERSION < 3 && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)')
    code.put_error_if_neg(self.pos, '__Pyx_init_sys_getdefaultencoding_params()')
    code.putln('#endif')
    code.putln('if (%s) {' % self.is_main_module_flag_cname())
    code.put_error_if_neg(self.pos, 'PyObject_SetAttr(%s, %s, %s)' % (env.module_cname, code.intern_identifier(EncodedString('__name__')), code.intern_identifier(EncodedString('__main__'))))
    code.putln('}')
    self.generate_module_import_setup(env, code)
    if Options.cache_builtins:
        code.putln('/*--- Builtin init code ---*/')
        code.put_error_if_neg(self.pos, '__Pyx_InitCachedBuiltins()')
    code.putln('/*--- Constants init code ---*/')
    code.put_error_if_neg(self.pos, '__Pyx_InitCachedConstants()')
    code.putln('/*--- Global type/function init code ---*/')
    with subfunction('Global init code') as inner_code:
        self.generate_global_init_code(env, inner_code)
    with subfunction('Variable export code') as inner_code:
        self.generate_c_variable_export_code(env, inner_code)
    with subfunction('Function export code') as inner_code:
        self.generate_c_function_export_code(env, inner_code)
    with subfunction('Type init code') as inner_code:
        self.generate_type_init_code(env, inner_code)
    with subfunction('Type import code') as inner_code:
        for module in imported_modules:
            self.generate_type_import_code_for_module(module, env, inner_code)
    with subfunction('Variable import code') as inner_code:
        for module in imported_modules:
            self.generate_c_variable_import_code_for_module(module, env, inner_code)
    with subfunction('Function import code') as inner_code:
        for module in imported_modules:
            self.specialize_fused_types(module)
            self.generate_c_function_import_code_for_module(module, env, inner_code)
    code.putln('/*--- Execution code ---*/')
    code.mark_pos(None)
    code.putln('#if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)')
    code.put_error_if_neg(self.pos, '__Pyx_patch_abc()')
    code.putln('#endif')
    if profile or linetrace:
        code.put_trace_call(header3, self.pos, nogil=not code.funcstate.gil_owned)
        code.funcstate.can_trace = True
    code.mark_pos(None)
    self.body.generate_execution_code(code)
    code.mark_pos(None)
    if profile or linetrace:
        code.funcstate.can_trace = False
        code.put_trace_return('Py_None', nogil=not code.funcstate.gil_owned)
    code.putln()
    code.putln('/*--- Wrapped vars code ---*/')
    self.generate_wrapped_entries_code(env, code)
    code.putln()
    if Options.generate_cleanup_code:
        code.globalstate.use_utility_code(UtilityCode.load_cached('RegisterModuleCleanup', 'ModuleSetupCode.c'))
        code.putln('if (__Pyx_RegisterCleanup()) %s' % code.error_goto(self.pos))
    code.put_goto(code.return_label)
    code.put_label(code.error_label)
    for cname, type in code.funcstate.all_managed_temps():
        code.put_xdecref(cname, type)
    code.putln('if (%s) {' % env.module_cname)
    code.putln('if (%s && stringtab_initialized) {' % env.module_dict_cname)
    code.put_add_traceback(EncodedString('init %s' % env.qualified_name))
    code.globalstate.use_utility_code(Nodes.traceback_utility_code)
    code.putln('}')
    code.putln('#if !CYTHON_USE_MODULE_STATE')
    code.put_decref_clear(env.module_cname, py_object_type, nanny=False, clear_before_decref=True)
    code.putln('#else')
    code.put_decref(env.module_cname, py_object_type, nanny=False)
    code.putln('if (pystate_addmodule_run) {')
    code.putln('PyObject *tp, *value, *tb;')
    code.putln('PyErr_Fetch(&tp, &value, &tb);')
    code.putln('PyState_RemoveModule(&%s);' % Naming.pymoduledef_cname)
    code.putln('PyErr_Restore(tp, value, tb);')
    code.putln('}')
    code.putln('#endif')
    code.putln('} else if (!PyErr_Occurred()) {')
    code.putln('PyErr_SetString(PyExc_ImportError, "init %s");' % env.qualified_name.as_c_string_literal()[1:-1])
    code.putln('}')
    code.put_label(code.return_label)
    code.put_finish_refcount_context()
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('return (%s != NULL) ? 0 : -1;' % env.module_cname)
    code.putln('#elif PY_MAJOR_VERSION >= 3')
    code.putln('return %s;' % env.module_cname)
    code.putln('#else')
    code.putln('return;')
    code.putln('#endif')
    code.putln('}')
    tempdecl_code.put_temp_declarations(code.funcstate)
    code.exit_cfunc_scope()