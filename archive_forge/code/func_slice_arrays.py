import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def slice_arrays(arrays, indices, contiguous=True):
    """Slices batches out of provided arrays (workaround for eager tensors).

  Unfortunately eager tensors don't have the same slicing behavior as
  Numpy arrays (they follow the same slicing behavior as symbolic TF tensors),
  hence we cannot use `generic_utils.slice_arrays` directly
  and we have to implement this workaround based on `concat`. This has a
  performance cost.

  Args:
    arrays: Single array or list of arrays.
    indices: List of indices in the array that should be included in the output
      batch.
    contiguous: Boolean flag indicating whether the indices are contiguous.

  Returns:
    Slice of data (either single array or list of arrays).
  """
    converted_to_list = False
    if not isinstance(arrays, list):
        converted_to_list = True
        arrays = [arrays]
    if any((tensor_util.is_tf_type(x) for x in arrays)):
        if not contiguous:
            entries = [[x[i:i + 1] for i in indices] for x in arrays]
            slices = [array_ops.concat(x, axis=0) for x in entries]
        else:
            slices = [x[indices[0]:indices[-1] + 1] for x in arrays]
    else:
        slices = generic_utils.slice_arrays(arrays, indices)
    if converted_to_list:
        slices = slices[0]
    return slices