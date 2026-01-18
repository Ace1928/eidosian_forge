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
def save_slices(filename: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shapes_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], data, name=None):
    """Saves input tensors slices to disk.

  This is like `Save` except that tensors can be listed in the saved file as being
  a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
  larger tensor and the slice that this tensor covers. `shapes_and_slices` must
  have as many elements as `tensor_names`.

  Elements of the `shapes_and_slices` input must either be:

  *  The empty string, in which case the corresponding tensor is
     saved normally.
  *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
     `dimI` are the dimensions of the larger tensor and `slice-spec`
     specifies what part is covered by the tensor to save.

  `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
  where each `sliceI` is either:

  *  The string `-` meaning that the slice covers all indices of this dimension
  *  `start,length` where `start` and `length` are integers.  In that
     case the slice covers `length` indices starting at `start`.

  See also `Save`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write the
      tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    shapes_and_slices: A `Tensor` of type `string`.
      Shape `[N]`.  The shapes and slice specifications to use when
      saving the tensors.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SaveSlices', name, filename, tensor_names, shapes_and_slices, data)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return save_slices_eager_fallback(filename, tensor_names, shapes_and_slices, data, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SaveSlices', filename=filename, tensor_names=tensor_names, shapes_and_slices=shapes_and_slices, data=data, name=name)
    return _op