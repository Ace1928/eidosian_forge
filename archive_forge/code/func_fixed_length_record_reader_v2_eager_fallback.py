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
def fixed_length_record_reader_v2_eager_fallback(record_bytes: int, header_bytes: int, footer_bytes: int, hop_bytes: int, container: str, shared_name: str, encoding: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    record_bytes = _execute.make_int(record_bytes, 'record_bytes')
    if header_bytes is None:
        header_bytes = 0
    header_bytes = _execute.make_int(header_bytes, 'header_bytes')
    if footer_bytes is None:
        footer_bytes = 0
    footer_bytes = _execute.make_int(footer_bytes, 'footer_bytes')
    if hop_bytes is None:
        hop_bytes = 0
    hop_bytes = _execute.make_int(hop_bytes, 'hop_bytes')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if encoding is None:
        encoding = ''
    encoding = _execute.make_str(encoding, 'encoding')
    _inputs_flat = []
    _attrs = ('header_bytes', header_bytes, 'record_bytes', record_bytes, 'footer_bytes', footer_bytes, 'hop_bytes', hop_bytes, 'container', container, 'shared_name', shared_name, 'encoding', encoding)
    _result = _execute.execute(b'FixedLengthRecordReaderV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FixedLengthRecordReaderV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result