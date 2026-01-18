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
def decode_proto_v2_eager_fallback(bytes: _atypes.TensorFuzzingAnnotation[_atypes.String], message_type: str, field_names, output_types, descriptor_source: str, message_format: str, sanitize: bool, name, ctx):
    message_type = _execute.make_str(message_type, 'message_type')
    if not isinstance(field_names, (list, tuple)):
        raise TypeError("Expected list for 'field_names' argument to 'decode_proto_v2' Op, not %r." % field_names)
    field_names = [_execute.make_str(_s, 'field_names') for _s in field_names]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'decode_proto_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if descriptor_source is None:
        descriptor_source = 'local://'
    descriptor_source = _execute.make_str(descriptor_source, 'descriptor_source')
    if message_format is None:
        message_format = 'binary'
    message_format = _execute.make_str(message_format, 'message_format')
    if sanitize is None:
        sanitize = False
    sanitize = _execute.make_bool(sanitize, 'sanitize')
    bytes = _ops.convert_to_tensor(bytes, _dtypes.string)
    _inputs_flat = [bytes]
    _attrs = ('message_type', message_type, 'field_names', field_names, 'output_types', output_types, 'descriptor_source', descriptor_source, 'message_format', message_format, 'sanitize', sanitize)
    _result = _execute.execute(b'DecodeProtoV2', len(output_types) + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodeProtoV2', _inputs_flat, _attrs, _result)
    _result = _result[:1] + [_result[1:]]
    _result = _DecodeProtoV2Output._make(_result)
    return _result