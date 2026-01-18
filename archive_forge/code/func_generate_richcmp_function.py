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
def generate_richcmp_function(self, scope, code):
    if scope.lookup_here('__richcmp__'):
        return
    richcmp_cfunc = scope.mangle_internal('tp_richcompare')
    code.putln('')
    code.putln('static PyObject *%s(PyObject *o1, PyObject *o2, int op) {' % richcmp_cfunc)
    code.putln('switch (op) {')
    class_scopes = []
    cls = scope.parent_type
    while cls is not None and (not cls.entry.visibility == 'extern'):
        class_scopes.append(cls.scope)
        cls = cls.scope.parent_type.base_type
    assert scope in class_scopes
    extern_parent = None
    if cls and cls.entry.visibility == 'extern':
        extern_parent = cls if cls.typeptr_cname else scope.parent_type.base_type
    total_ordering = 'total_ordering' in scope.directives
    comp_entry = {}
    for cmp_method in TypeSlots.richcmp_special_methods:
        for class_scope in class_scopes:
            entry = class_scope.lookup_here(cmp_method)
            if entry is not None:
                comp_entry[cmp_method] = entry
                break
    if total_ordering:
        comp_names = [from_name for from_name, to_name in TOTAL_ORDERING if from_name in comp_entry]
        if not comp_names:
            if '__eq__' not in comp_entry and '__ne__' not in comp_entry:
                warning(scope.parent_type.pos, 'total_ordering directive used, but no comparison and equality methods defined')
            else:
                warning(scope.parent_type.pos, 'total_ordering directive used, but no comparison methods defined')
            total_ordering = False
        else:
            if '__eq__' not in comp_entry and '__ne__' not in comp_entry:
                warning(scope.parent_type.pos, 'total_ordering directive used, but no equality method defined')
                total_ordering = False
            ordering_source = max(comp_names)
    for cmp_method in TypeSlots.richcmp_special_methods:
        cmp_type = cmp_method.strip('_').upper()
        entry = comp_entry.get(cmp_method)
        if entry is None and (not total_ordering or cmp_type in ('NE', 'EQ')):
            continue
        code.putln('case Py_%s: {' % cmp_type)
        if entry is None:
            assert total_ordering
            invert_comp, comp_op, invert_equals = TOTAL_ORDERING[ordering_source, cmp_method]
            code.putln('PyObject *ret;')
            code.putln('ret = %s(o1, o2);' % comp_entry[ordering_source].func_cname)
            code.putln('if (likely(ret && ret != Py_NotImplemented)) {')
            code.putln('int order_res = __Pyx_PyObject_IsTrue(ret);')
            code.putln('Py_DECREF(ret);')
            code.putln('if (unlikely(order_res < 0)) return NULL;')
            if invert_equals is not None:
                if comp_op == '&&':
                    code.putln('if (%s order_res) {' % ('!!' if invert_comp else '!'))
                    code.putln('ret = __Pyx_NewRef(Py_False);')
                    code.putln('} else {')
                elif comp_op == '||':
                    code.putln('if (%s order_res) {' % ('!' if invert_comp else ''))
                    code.putln('ret = __Pyx_NewRef(Py_True);')
                    code.putln('} else {')
                else:
                    raise AssertionError('Unknown op %s' % (comp_op,))
                if '__eq__' in comp_entry:
                    eq_func = '__eq__'
                else:
                    eq_func = '__ne__'
                    invert_equals = not invert_equals
                code.putln('ret = %s(o1, o2);' % comp_entry[eq_func].func_cname)
                code.putln('if (likely(ret && ret != Py_NotImplemented)) {')
                code.putln('int eq_res = __Pyx_PyObject_IsTrue(ret);')
                code.putln('Py_DECREF(ret);')
                code.putln('if (unlikely(eq_res < 0)) return NULL;')
                if invert_equals:
                    code.putln('ret = eq_res ? Py_False : Py_True;')
                else:
                    code.putln('ret = eq_res ? Py_True : Py_False;')
                code.putln('Py_INCREF(ret);')
                code.putln('}')
                code.putln('}')
            else:
                if invert_comp:
                    code.putln('ret = order_res ? Py_False : Py_True;')
                else:
                    code.putln('ret = order_res ? Py_True : Py_False;')
                code.putln('Py_INCREF(ret);')
            code.putln('}')
            code.putln('return ret;')
        else:
            code.putln('return %s(o1, o2);' % entry.func_cname)
        code.putln('}')
    if '__eq__' in comp_entry and '__ne__' not in comp_entry and (not extern_parent):
        code.putln('case Py_NE: {')
        code.putln('PyObject *ret;')
        code.putln('ret = %s(o1, o2);' % comp_entry['__eq__'].func_cname)
        code.putln('if (likely(ret && ret != Py_NotImplemented)) {')
        code.putln('int b = __Pyx_PyObject_IsTrue(ret);')
        code.putln('Py_DECREF(ret);')
        code.putln('if (unlikely(b < 0)) return NULL;')
        code.putln('ret = (b) ? Py_False : Py_True;')
        code.putln('Py_INCREF(ret);')
        code.putln('}')
        code.putln('return ret;')
        code.putln('}')
    code.putln('default: {')
    if extern_parent and extern_parent.typeptr_cname:
        code.putln('if (likely(%s->tp_richcompare)) return %s->tp_richcompare(o1, o2, op);' % (extern_parent.typeptr_cname, extern_parent.typeptr_cname))
    code.putln('return __Pyx_NewRef(Py_NotImplemented);')
    code.putln('}')
    code.putln('}')
    code.putln('}')