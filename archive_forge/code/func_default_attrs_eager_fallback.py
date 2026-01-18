import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def default_attrs_eager_fallback(string_val: str, string_list_val, int_val: int, int_list_val, float_val: float, float_list_val, bool_val: bool, bool_list_val, type_val: TV_DefaultAttrs_type_val, type_list_val, shape_val, shape_list_val, tensor_val, tensor_list_val, name, ctx):
    if string_val is None:
        string_val = 'abc'
    string_val = _execute.make_str(string_val, 'string_val')
    if string_list_val is None:
        string_list_val = ['abc', '']
    if not isinstance(string_list_val, (list, tuple)):
        raise TypeError("Expected list for 'string_list_val' argument to 'default_attrs' Op, not %r." % string_list_val)
    string_list_val = [_execute.make_str(_s, 'string_list_val') for _s in string_list_val]
    if int_val is None:
        int_val = 123
    int_val = _execute.make_int(int_val, 'int_val')
    if int_list_val is None:
        int_list_val = [1, 2, 3]
    if not isinstance(int_list_val, (list, tuple)):
        raise TypeError("Expected list for 'int_list_val' argument to 'default_attrs' Op, not %r." % int_list_val)
    int_list_val = [_execute.make_int(_i, 'int_list_val') for _i in int_list_val]
    if float_val is None:
        float_val = 10
    float_val = _execute.make_float(float_val, 'float_val')
    if float_list_val is None:
        float_list_val = [10]
    if not isinstance(float_list_val, (list, tuple)):
        raise TypeError("Expected list for 'float_list_val' argument to 'default_attrs' Op, not %r." % float_list_val)
    float_list_val = [_execute.make_float(_f, 'float_list_val') for _f in float_list_val]
    if bool_val is None:
        bool_val = True
    bool_val = _execute.make_bool(bool_val, 'bool_val')
    if bool_list_val is None:
        bool_list_val = [True, False]
    if not isinstance(bool_list_val, (list, tuple)):
        raise TypeError("Expected list for 'bool_list_val' argument to 'default_attrs' Op, not %r." % bool_list_val)
    bool_list_val = [_execute.make_bool(_b, 'bool_list_val') for _b in bool_list_val]
    if type_val is None:
        type_val = _dtypes.int32
    type_val = _execute.make_type(type_val, 'type_val')
    if type_list_val is None:
        type_list_val = [_dtypes.int32, _dtypes.float32]
    if not isinstance(type_list_val, (list, tuple)):
        raise TypeError("Expected list for 'type_list_val' argument to 'default_attrs' Op, not %r." % type_list_val)
    type_list_val = [_execute.make_type(_t, 'type_list_val') for _t in type_list_val]
    if shape_val is None:
        shape_val = [2, 1]
    shape_val = _execute.make_shape(shape_val, 'shape_val')
    if shape_list_val is None:
        shape_list_val = [[], [1]]
    if not isinstance(shape_list_val, (list, tuple)):
        raise TypeError("Expected list for 'shape_list_val' argument to 'default_attrs' Op, not %r." % shape_list_val)
    shape_list_val = [_execute.make_shape(_s, 'shape_list_val') for _s in shape_list_val]
    if tensor_val is None:
        tensor_val = _execute.make_tensor('dtype: DT_INT32 tensor_shape { } int_val: 1 ', 'tensor_val')
    tensor_val = _execute.make_tensor(tensor_val, 'tensor_val')
    if tensor_list_val is None:
        tensor_list_val = [_execute.make_tensor(_pb, 'tensor_list_val') for _pb in ('dtype: DT_INT32 tensor_shape { } int_val: 1 ',)]
    if not isinstance(tensor_list_val, (list, tuple)):
        raise TypeError("Expected list for 'tensor_list_val' argument to 'default_attrs' Op, not %r." % tensor_list_val)
    tensor_list_val = [_execute.make_tensor(_t, 'tensor_list_val') for _t in tensor_list_val]
    _inputs_flat = []
    _attrs = ('string_val', string_val, 'string_list_val', string_list_val, 'int_val', int_val, 'int_list_val', int_list_val, 'float_val', float_val, 'float_list_val', float_list_val, 'bool_val', bool_val, 'bool_list_val', bool_list_val, 'type_val', type_val, 'type_list_val', type_list_val, 'shape_val', shape_val, 'shape_list_val', shape_list_val, 'tensor_val', tensor_val, 'tensor_list_val', tensor_list_val)
    _result = _execute.execute(b'DefaultAttrs', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result