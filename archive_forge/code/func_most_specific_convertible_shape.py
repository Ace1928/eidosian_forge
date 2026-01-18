from . import compat, dtypes
from tensorboard.compat.proto import tensor_shape_pb2
def most_specific_convertible_shape(self, other):
    """Returns the most specific TensorShape convertible with `self` and
        `other`.

        * TensorShape([None, 1]) is the most specific TensorShape convertible with
          both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
          TensorShape(None) is also convertible with above mentioned TensorShapes.

        * TensorShape([1, 2, 3]) is the most specific TensorShape convertible with
          both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
          less specific TensorShapes convertible with above mentioned TensorShapes,
          e.g. TensorShape([1, 2, None]), TensorShape(None).

        Args:
          other: Another `TensorShape`.

        Returns:
          A `TensorShape` which is the most specific convertible shape of `self`
          and `other`.
        """
    other = as_shape(other)
    if self._dims is None or other.dims is None or self.ndims != other.ndims:
        return unknown_shape()
    dims = [Dimension(None)] * self.ndims
    for i, (d1, d2) in enumerate(zip(self._dims, other.dims)):
        if d1 is not None and d2 is not None and (d1 == d2):
            dims[i] = d1
    return TensorShape(dims)