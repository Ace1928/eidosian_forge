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
def generate_new_function(self, scope, code, cclass_entry):
    tp_slot = TypeSlots.ConstructorSlot('tp_new', '__cinit__')
    slot_func = scope.mangle_internal('tp_new')
    if tp_slot.slot_code(scope) != slot_func:
        return
    type = scope.parent_type
    base_type = type.base_type
    have_entries, (py_attrs, py_buffers, memoryview_slices) = scope.get_refcounted_entries()
    is_final_type = scope.parent_type.is_final_type
    if scope.is_internal:
        py_attrs = []
    cpp_constructable_attrs = [entry for entry in scope.var_entries if entry.type.needs_cpp_construction]
    cinit_func_entry = scope.lookup_here('__cinit__')
    if cinit_func_entry and (not cinit_func_entry.is_special):
        cinit_func_entry = None
    if base_type or (cinit_func_entry and (not cinit_func_entry.trivial_signature)):
        unused_marker = ''
    else:
        unused_marker = 'CYTHON_UNUSED '
    if base_type:
        freelist_size = 0
    else:
        freelist_size = scope.directives.get('freelist', 0)
    freelist_name = scope.mangle_internal(Naming.freelist_name)
    freecount_name = scope.mangle_internal(Naming.freecount_name)
    decls = code.globalstate['decls']
    decls.putln('static PyObject *%s(PyTypeObject *t, PyObject *a, PyObject *k); /*proto*/' % slot_func)
    code.putln('')
    if freelist_size:
        code.putln('#if CYTHON_USE_FREELISTS')
        code.putln('static %s[%d];' % (scope.parent_type.declaration_code(freelist_name), freelist_size))
        code.putln('static int %s = 0;' % freecount_name)
        code.putln('#endif')
        code.putln('')
    code.putln('static PyObject *%s(PyTypeObject *t, %sPyObject *a, %sPyObject *k) {' % (slot_func, unused_marker, unused_marker))
    need_self_cast = type.vtabslot_cname or (py_buffers or memoryview_slices or py_attrs) or cpp_constructable_attrs
    if need_self_cast:
        code.putln('%s;' % scope.parent_type.declaration_code('p'))
    if base_type:
        tp_new = TypeSlots.get_base_slot_function(scope, tp_slot)
        if tp_new is None:
            tp_new = '__Pyx_PyType_GetSlot(%s, tp_new, newfunc)' % base_type.typeptr_cname
        code.putln('PyObject *o = %s(t, a, k);' % tp_new)
    else:
        code.putln('PyObject *o;')
        code.putln('#if CYTHON_COMPILING_IN_LIMITED_API')
        code.putln('allocfunc alloc_func = (allocfunc)PyType_GetSlot(t, Py_tp_alloc);')
        code.putln('o = alloc_func(t, 0);')
        code.putln('#else')
        if freelist_size:
            code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
            if is_final_type:
                type_safety_check = ''
            else:
                type_safety_check = ' & (int)(!__Pyx_PyType_HasFeature(t, (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)))'
            obj_struct = type.declaration_code('', deref=True)
            code.putln('#if CYTHON_USE_FREELISTS')
            code.putln('if (likely((int)(%s > 0) & (int)(t->tp_basicsize == sizeof(%s))%s)) {' % (freecount_name, obj_struct, type_safety_check))
            code.putln('o = (PyObject*)%s[--%s];' % (freelist_name, freecount_name))
            code.putln('memset(o, 0, sizeof(%s));' % obj_struct)
            code.putln('(void) PyObject_INIT(o, t);')
            if scope.needs_gc():
                code.putln('PyObject_GC_Track(o);')
            code.putln('} else')
            code.putln('#endif')
            code.putln('{')
        if not is_final_type:
            code.putln('if (likely(!__Pyx_PyType_HasFeature(t, Py_TPFLAGS_IS_ABSTRACT))) {')
        code.putln('o = (*t->tp_alloc)(t, 0);')
        if not is_final_type:
            code.putln('} else {')
            code.putln('o = (PyObject *) PyBaseObject_Type.tp_new(t, %s, 0);' % Naming.empty_tuple)
            code.putln('}')
    code.putln('if (unlikely(!o)) return 0;')
    if freelist_size and (not base_type):
        code.putln('}')
    if not base_type:
        code.putln('#endif')
    if need_self_cast:
        code.putln('p = %s;' % type.cast_code('o'))
    needs_error_cleanup = False
    if type.vtabslot_cname:
        vtab_base_type = type
        while vtab_base_type.base_type and vtab_base_type.base_type.vtabstruct_cname:
            vtab_base_type = vtab_base_type.base_type
        if vtab_base_type is not type:
            struct_type_cast = '(struct %s*)' % vtab_base_type.vtabstruct_cname
        else:
            struct_type_cast = ''
        code.putln('p->%s = %s%s;' % (type.vtabslot_cname, struct_type_cast, type.vtabptr_cname))
    for entry in cpp_constructable_attrs:
        if entry.is_cpp_optional:
            decl_code = entry.type.cpp_optional_declaration_code('')
        else:
            decl_code = entry.type.empty_declaration_code()
        code.putln('new((void*)&(p->%s)) %s();' % (entry.cname, decl_code))
    for entry in py_attrs:
        if entry.name == '__dict__':
            needs_error_cleanup = True
            code.put('p->%s = PyDict_New(); if (unlikely(!p->%s)) goto bad;' % (entry.cname, entry.cname))
        else:
            code.put_init_var_to_py_none(entry, 'p->%s', nanny=False)
    for entry in memoryview_slices:
        code.putln('p->%s.data = NULL;' % entry.cname)
        code.putln('p->%s.memview = NULL;' % entry.cname)
    for entry in py_buffers:
        code.putln('p->%s.obj = NULL;' % entry.cname)
    if cclass_entry.cname == '__pyx_memoryviewslice':
        code.putln('p->from_slice.memview = NULL;')
    if cinit_func_entry:
        if cinit_func_entry.trivial_signature:
            cinit_args = 'o, %s, NULL' % Naming.empty_tuple
        else:
            cinit_args = 'o, a, k'
        needs_error_cleanup = True
        code.putln('if (unlikely(%s(%s) < 0)) goto bad;' % (cinit_func_entry.func_cname, cinit_args))
    code.putln('return o;')
    if needs_error_cleanup:
        code.putln('bad:')
        code.put_decref_clear('o', py_object_type, nanny=False)
        code.putln('return NULL;')
    code.putln('}')