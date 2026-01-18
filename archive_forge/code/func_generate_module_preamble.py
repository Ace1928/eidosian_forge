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
def generate_module_preamble(self, env, options, cimported_modules, metadata, code):
    code.put_generated_by()
    if metadata:
        code.putln('/* BEGIN: Cython Metadata')
        code.putln(json.dumps(metadata, indent=4, sort_keys=True))
        code.putln('END: Cython Metadata */')
        code.putln('')
    code.putln('#ifndef PY_SSIZE_T_CLEAN')
    code.putln('#define PY_SSIZE_T_CLEAN')
    code.putln('#endif /* PY_SSIZE_T_CLEAN */')
    self._put_setup_code(code, 'InitLimitedAPI')
    for inc in sorted(env.c_includes.values(), key=IncludeCode.sortkey):
        if inc.location == inc.INITIAL:
            inc.write(code)
    code.putln('#ifndef Py_PYTHON_H')
    code.putln('    #error Python headers needed to compile C extensions, please install development version of Python.')
    code.putln('#elif PY_VERSION_HEX < 0x02070000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)')
    code.putln('    #error Cython requires Python 2.7+ or Python 3.3+.')
    code.putln('#else')
    code.globalstate['end'].putln('#endif /* Py_PYTHON_H */')
    from .. import __version__
    code.putln('#if defined(CYTHON_LIMITED_API) && CYTHON_LIMITED_API')
    code.putln('#define __PYX_EXTRA_ABI_MODULE_NAME "limited"')
    code.putln('#else')
    code.putln('#define __PYX_EXTRA_ABI_MODULE_NAME ""')
    code.putln('#endif')
    code.putln('#define CYTHON_ABI "%s" __PYX_EXTRA_ABI_MODULE_NAME' % __version__.replace('.', '_'))
    code.putln('#define __PYX_ABI_MODULE_NAME "_cython_" CYTHON_ABI')
    code.putln('#define __PYX_TYPE_MODULE_PREFIX __PYX_ABI_MODULE_NAME "."')
    code.putln('#define CYTHON_HEX_VERSION %s' % build_hex_version(__version__))
    code.putln('#define CYTHON_FUTURE_DIVISION %d' % (Future.division in env.context.future_directives))
    self._put_setup_code(code, 'CModulePreamble')
    if env.context.options.cplus:
        self._put_setup_code(code, 'CppInitCode')
    else:
        self._put_setup_code(code, 'CInitCode')
    self._put_setup_code(code, 'PythonCompatibility')
    self._put_setup_code(code, 'MathInitCode')
    if options.c_line_in_traceback:
        cinfo = '%s = %s; ' % (Naming.clineno_cname, Naming.line_c_macro)
    else:
        cinfo = ''
    code.putln('#define __PYX_MARK_ERR_POS(f_index, lineno) \\')
    code.putln('    { %s = %s[f_index]; (void)%s; %s = lineno; (void)%s; %s (void)%s; }' % (Naming.filename_cname, Naming.filetable_cname, Naming.filename_cname, Naming.lineno_cname, Naming.lineno_cname, cinfo, Naming.clineno_cname))
    code.putln('#define __PYX_ERR(f_index, lineno, Ln_error) \\')
    code.putln('    { __PYX_MARK_ERR_POS(f_index, lineno) goto Ln_error; }')
    code.putln('')
    self.generate_extern_c_macro_definition(code, env.is_cpp())
    code.putln('')
    code.putln('#define %s' % self.api_name(Naming.h_guard_prefix, env))
    code.putln('#define %s' % self.api_name(Naming.api_guard_prefix, env))
    code.putln('/* Early includes */')
    self.generate_includes(env, cimported_modules, code, late=False)
    code.putln('')
    code.putln('#if defined(PYREX_WITHOUT_ASSERTIONS) && !defined(CYTHON_WITHOUT_ASSERTIONS)')
    code.putln('#define CYTHON_WITHOUT_ASSERTIONS')
    code.putln('#endif')
    code.putln('')
    if env.directives['ccomplex']:
        code.putln('')
        code.putln('#if !defined(CYTHON_CCOMPLEX)')
        code.putln('#define CYTHON_CCOMPLEX 1')
        code.putln('#endif')
        code.putln('')
    code.put(UtilityCode.load_as_string('UtilityFunctionPredeclarations', 'ModuleSetupCode.c')[0])
    c_string_type = env.directives['c_string_type']
    c_string_encoding = env.directives['c_string_encoding']
    if c_string_type not in ('bytes', 'bytearray') and (not c_string_encoding):
        error(self.pos, 'a default encoding must be provided if c_string_type is not a byte type')
    code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_ASCII %s' % int(c_string_encoding == 'ascii'))
    code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_UTF8 %s' % int(c_string_encoding.replace('-', '').lower() == 'utf8'))
    if c_string_encoding == 'default':
        code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT 1')
    else:
        code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT (PY_MAJOR_VERSION >= 3 && __PYX_DEFAULT_STRING_ENCODING_IS_UTF8)')
        code.putln('#define __PYX_DEFAULT_STRING_ENCODING "%s"' % c_string_encoding)
    if c_string_type == 'bytearray':
        c_string_func_name = 'ByteArray'
    else:
        c_string_func_name = c_string_type.title()
    code.putln('#define __Pyx_PyObject_FromString __Pyx_Py%s_FromString' % c_string_func_name)
    code.putln('#define __Pyx_PyObject_FromStringAndSize __Pyx_Py%s_FromStringAndSize' % c_string_func_name)
    code.put(UtilityCode.load_as_string('TypeConversions', 'TypeConversion.c')[0])
    env.use_utility_code(UtilityCode.load_cached('FormatTypeName', 'ObjectHandling.c'))
    PyrexTypes.c_long_type.create_to_py_utility_code(env)
    PyrexTypes.c_long_type.create_from_py_utility_code(env)
    PyrexTypes.c_int_type.create_from_py_utility_code(env)
    code.put(Nodes.branch_prediction_macros)
    code.putln('static CYTHON_INLINE void __Pyx_pretend_to_initialize(void* ptr) { (void)ptr; }')
    code.putln('')
    code.putln('#if !CYTHON_USE_MODULE_STATE')
    code.putln('static PyObject *%s = NULL;' % env.module_cname)
    if Options.pre_import is not None:
        code.putln('static PyObject *%s;' % Naming.preimport_cname)
    code.putln('#endif')
    code.putln('static int %s;' % Naming.lineno_cname)
    code.putln('static int %s = 0;' % Naming.clineno_cname)
    code.putln('static const char * %s = %s;' % (Naming.cfilenm_cname, Naming.file_c_macro))
    code.putln('static const char *%s;' % Naming.filename_cname)
    env.use_utility_code(UtilityCode.load_cached('FastTypeChecks', 'ModuleSetupCode.c'))
    if has_np_pythran(env):
        env.use_utility_code(UtilityCode.load_cached('PythranConversion', 'CppSupport.cpp'))