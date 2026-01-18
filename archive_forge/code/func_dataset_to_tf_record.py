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
def dataset_to_tf_record(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], filename: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None):
    """Writes the given dataset to the given file using the TFRecord format.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to write.
    filename: A `Tensor` of type `string`.
      A scalar string tensor representing the filename to use.
    compression_type: A `Tensor` of type `string`.
      A scalar string tensor containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DatasetToTFRecord', name, input_dataset, filename, compression_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dataset_to_tf_record_eager_fallback(input_dataset, filename, compression_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DatasetToTFRecord', input_dataset=input_dataset, filename=filename, compression_type=compression_type, name=name)
    return _op