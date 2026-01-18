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
def dataset_cardinality(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], cardinality_options: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Returns the cardinality of `input_dataset`.

  Returns the cardinality of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return cardinality for.
    cardinality_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DatasetCardinality', name, input_dataset, 'cardinality_options', cardinality_options)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dataset_cardinality_eager_fallback(input_dataset, cardinality_options=cardinality_options, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if cardinality_options is None:
        cardinality_options = ''
    cardinality_options = _execute.make_str(cardinality_options, 'cardinality_options')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DatasetCardinality', input_dataset=input_dataset, cardinality_options=cardinality_options, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('cardinality_options', _op.get_attr('cardinality_options'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DatasetCardinality', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result