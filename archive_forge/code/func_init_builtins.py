from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def init_builtins():
    init_builtin_structs()
    init_builtin_types()
    init_builtin_funcs()
    entry = builtin_scope.declare_var('__debug__', PyrexTypes.c_const_type(PyrexTypes.c_bint_type), pos=None, cname='__pyx_assertions_enabled()', is_cdef=True)
    entry.utility_code = UtilityCode.load_cached('AssertionsEnabled', 'Exceptions.c')
    global type_type, list_type, tuple_type, dict_type, set_type, frozenset_type, slice_type
    global bytes_type, str_type, unicode_type, basestring_type, bytearray_type
    global float_type, int_type, long_type, bool_type, complex_type
    global memoryview_type, py_buffer_type
    global sequence_types
    type_type = builtin_scope.lookup('type').type
    list_type = builtin_scope.lookup('list').type
    tuple_type = builtin_scope.lookup('tuple').type
    dict_type = builtin_scope.lookup('dict').type
    set_type = builtin_scope.lookup('set').type
    frozenset_type = builtin_scope.lookup('frozenset').type
    slice_type = builtin_scope.lookup('slice').type
    bytes_type = builtin_scope.lookup('bytes').type
    str_type = builtin_scope.lookup('str').type
    unicode_type = builtin_scope.lookup('unicode').type
    basestring_type = builtin_scope.lookup('basestring').type
    bytearray_type = builtin_scope.lookup('bytearray').type
    memoryview_type = builtin_scope.lookup('memoryview').type
    float_type = builtin_scope.lookup('float').type
    int_type = builtin_scope.lookup('int').type
    long_type = builtin_scope.lookup('long').type
    bool_type = builtin_scope.lookup('bool').type
    complex_type = builtin_scope.lookup('complex').type
    sequence_types = (list_type, tuple_type, bytes_type, str_type, unicode_type, basestring_type, bytearray_type, memoryview_type)
    bool_type.equivalent_type = PyrexTypes.c_bint_type
    PyrexTypes.c_bint_type.equivalent_type = bool_type
    float_type.equivalent_type = PyrexTypes.c_double_type
    PyrexTypes.c_double_type.equivalent_type = float_type
    complex_type.equivalent_type = PyrexTypes.c_double_complex_type
    PyrexTypes.c_double_complex_type.equivalent_type = complex_type
    py_buffer_type = builtin_scope.lookup('Py_buffer').type