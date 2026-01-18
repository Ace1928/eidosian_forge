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
def mirror_pad_grad(input: _atypes.TensorFuzzingAnnotation[TV_MirrorPadGrad_T], paddings: _atypes.TensorFuzzingAnnotation[TV_MirrorPadGrad_Tpaddings], mode: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MirrorPadGrad_T]:
    """Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

  This operation folds the padded areas of `input` by `MirrorPad` according to the
  `paddings` you specify. `paddings` must be the same as `paddings` argument
  given to the corresponding `MirrorPad` op.

  The folded size of each dimension D of the output is:

  `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
  # 'paddings' is [[0, 1]], [0, 1]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[ 1,  5]
                        [11, 28]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be folded.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      The mode used in the `MirrorPad` op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MirrorPadGrad', name, input, paddings, 'mode', mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mirror_pad_grad_eager_fallback(input, paddings, mode=mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    mode = _execute.make_str(mode, 'mode')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MirrorPadGrad', input=input, paddings=paddings, mode=mode, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tpaddings', _op._get_attr_type('Tpaddings'), 'mode', _op.get_attr('mode'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MirrorPadGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result