from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(bincount_ops.bincount)
def bincount(arr: ragged_tensor.RaggedTensor, weights=None, minlength=None, maxlength=None, dtype=dtypes.int32, name=None, axis=None, binary_output=False):
    """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.
  If `weights` are non-None, then index `i` of the output stores the sum of the
  value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  ```python
  values = tf.constant([1,1,2,3,2,4,4,5])
  tf.math.bincount(values) #[0 2 2 1 2 1]
  ```
  Vector length = Maximum element in vector `values` is 5. Adding 1, which is 6
                  will be the vector length.

  Each bin value in the output indicates number of occurrences of the particular
  index. Here, index 1 in output has a value 2. This indicates value 1 occurs
  two times in `values`.

  ```python
  values = tf.constant([1,1,2,3,2,4,4,5])
  weights = tf.constant([1,5,0,1,0,5,4,5])
  tf.math.bincount(values, weights=weights) #[0 6 0 1 9 5]
  ```
  Bin will be incremented by the corresponding weight instead of 1.
  Here, index 1 in output has a value 6. This is the summation of weights
  corresponding to the value in `values`.

  **Bin-counting on a certain axis**

  This example takes a 2 dimensional input and returns a `Tensor` with
  bincounting on each sample.

  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[1, 1, 1, 1],
           [2, 1, 1, 0]], dtype=int32)>


  **Bin-counting with binary_output**

  This example gives binary output instead of counting the occurrence.

  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1, binary_output=True)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[1, 1, 1, 1],
           [1, 1, 1, 0]], dtype=int32)>

  Args:
    arr: A RaggedTensor whose values should be counted.
      These tensors must have a rank of 2 if `axis=-1`.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    dtype: If `weights` is None, determines the type of the output bins.
    name: A name scope for the associated operations (optional).
    axis: The axis to slice over. Axes at and below `axis` will be flattened
      before bin counting. Currently, only `0`, and `-1` are supported. If None,
      all axes will be flattened (identical to passing `0`).
    binary_output: If True, this op will output 1 instead of the number of times
      a token appears (equivalent to one_hot + reduce_any instead of one_hot +
      reduce_add). Defaults to False.

  Returns:
    A vector with the same dtype as `weights` or the given `dtype`. The bin
    values.

  Raises:
    `InvalidArgumentError` if negative values are provided as an input.

  """
    name = 'bincount' if name is None else name
    with ops.name_scope(name):
        arr = ragged_tensor.convert_to_tensor_or_ragged_tensor(arr, name='arr')
        if weights is not None:
            if not isinstance(weights, sparse_tensor.SparseTensor):
                weights = ragged_tensor.convert_to_tensor_or_ragged_tensor(weights, name='weights')
        if weights is not None and binary_output:
            raise ValueError('Arguments `binary_output` and `weights` are mutually exclusive. Please specify only one.')
        if not arr.dtype.is_integer:
            arr = math_ops.cast(arr, dtypes.int32)
        if axis is None:
            axis = 0
        if axis not in [0, -1]:
            raise ValueError(f'Unsupported value for argument axis={axis}. Only 0 and -1 are currently supported.')
        array_is_nonempty = array_ops.size(arr) > 0
        output_size = math_ops.cast(array_is_nonempty, arr.dtype) * (math_ops.reduce_max(arr) + 1)
        if minlength is not None:
            minlength = ops.convert_to_tensor(minlength, name='minlength', dtype=arr.dtype)
            output_size = gen_math_ops.maximum(minlength, output_size)
        if maxlength is not None:
            maxlength = ops.convert_to_tensor(maxlength, name='maxlength', dtype=arr.dtype)
            output_size = gen_math_ops.minimum(maxlength, output_size)
        if axis == 0:
            while isinstance(arr, ragged_tensor.RaggedTensor):
                if weights is not None:
                    weights = validate_ragged_weights(arr, weights, dtype)
                arr = arr.values
        if isinstance(arr, ragged_tensor.RaggedTensor):
            weights = validate_ragged_weights(arr, weights, dtype)
            return gen_math_ops.ragged_bincount(splits=arr.row_splits, values=arr.values, size=output_size, weights=weights, binary_output=binary_output)
        else:
            weights = bincount_ops.validate_dense_weights(arr, weights, dtype)
            return gen_math_ops.dense_bincount(input=arr, size=output_size, weights=weights, binary_output=binary_output)