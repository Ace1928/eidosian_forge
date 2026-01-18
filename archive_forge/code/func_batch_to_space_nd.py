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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export(v1=['batch_to_space_nd', 'manip.batch_to_space_nd'])
@deprecated_endpoints('batch_to_space_nd', 'manip.batch_to_space_nd')
def batch_to_space_nd(input: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_T], block_shape: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_Tblock_shape], crops: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_Tcrops], name=None) -> _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_T]:
    """BatchToSpace for N-D tensors of type T.

  This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
  `block_shape + [batch]`, interleaves these blocks back into the grid defined by
  the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
  the input.  The spatial dimensions of this intermediate result are then
  optionally cropped according to `crops` to produce the output.  This is the
  reverse of SpaceToBatch.  See below for a precise description.

  Args:
    input: A `Tensor`.
      N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
      where spatial_shape has M dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[M]`, all values must be >= 1.
    crops: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D with shape `[M, 2]`, all values must be >= 0.
        `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
        dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
        required that
        `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.

      This operation is equivalent to the following steps:

      1. Reshape `input` to `reshaped` of shape:
           [block_shape[0], ..., block_shape[M-1],
            batch / prod(block_shape),
            input_shape[1], ..., input_shape[N-1]]

      2. Permute dimensions of `reshaped` to produce `permuted` of shape
           [batch / prod(block_shape),

            input_shape[1], block_shape[0],
            ...,
            input_shape[M], block_shape[M-1],

            input_shape[M+1], ..., input_shape[N-1]]

      3. Reshape `permuted` to produce `reshaped_permuted` of shape
           [batch / prod(block_shape),

            input_shape[1] * block_shape[0],
            ...,
            input_shape[M] * block_shape[M-1],

            input_shape[M+1],
            ...,
            input_shape[N-1]]

      4. Crop the start and end of dimensions `[1, ..., M]` of
         `reshaped_permuted` according to `crops` to produce the output of shape:
           [batch / prod(block_shape),

            input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
            ...,
            input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],

            input_shape[M+1], ..., input_shape[N-1]]

      Some examples:

      (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
      ```

      The output tensor has shape `[1, 2, 2, 1]` and value:

      ```
      x = [[[[1], [2]], [[3], [4]]]]
      ```

      (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
      ```

      The output tensor has shape `[1, 2, 2, 3]` and value:

      ```
      x = [[[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]]]
      ```

      (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1], [3]], [[9], [11]]],
           [[[2], [4]], [[10], [12]]],
           [[[5], [7]], [[13], [15]]],
           [[[6], [8]], [[14], [16]]]]
      ```

      The output tensor has shape `[1, 4, 4, 1]` and value:

      ```
      x = [[[[1],   [2],  [3],  [4]],
           [[5],   [6],  [7],  [8]],
           [[9],  [10], [11],  [12]],
           [[13], [14], [15],  [16]]]]
      ```

      (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
          `crops = [[0, 0], [2, 0]]`:

      ```
      x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
           [[[0], [2], [4]]], [[[0], [10], [12]]],
           [[[0], [5], [7]]], [[[0], [13], [15]]],
           [[[0], [6], [8]]], [[[0], [14], [16]]]]
      ```

      The output tensor has shape `[2, 2, 4, 1]` and value:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]]],
           [[[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchToSpaceND', name, input, block_shape, crops)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_batch_to_space_nd((input, block_shape, crops, name), None)
            if _result is not NotImplemented:
                return _result
            return batch_to_space_nd_eager_fallback(input, block_shape, crops, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(batch_to_space_nd, (), dict(input=input, block_shape=block_shape, crops=crops, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_batch_to_space_nd((input, block_shape, crops, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchToSpaceND', input=input, block_shape=block_shape, crops=crops, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(batch_to_space_nd, (), dict(input=input, block_shape=block_shape, crops=crops, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tblock_shape', _op._get_attr_type('Tblock_shape'), 'Tcrops', _op._get_attr_type('Tcrops'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchToSpaceND', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result