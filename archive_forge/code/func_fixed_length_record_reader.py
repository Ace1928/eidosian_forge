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
def fixed_length_record_reader(record_bytes: int, header_bytes: int=0, footer_bytes: int=0, hop_bytes: int=0, container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`. Number of bytes in the record.
    header_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the header, defaults to 0.
    footer_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the footer, defaults to 0.
    hop_bytes: An optional `int`. Defaults to `0`.
      Number of bytes to hop before each read. Default of 0 means using
      record_bytes.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("fixed_length_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FixedLengthRecordReader', record_bytes=record_bytes, header_bytes=header_bytes, footer_bytes=footer_bytes, hop_bytes=hop_bytes, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('header_bytes', _op._get_attr_int('header_bytes'), 'record_bytes', _op._get_attr_int('record_bytes'), 'footer_bytes', _op._get_attr_int('footer_bytes'), 'hop_bytes', _op._get_attr_int('hop_bytes'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FixedLengthRecordReader', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result