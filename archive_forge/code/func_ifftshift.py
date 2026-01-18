import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.ifftshift')
@dispatch.add_dispatch_support
def ifftshift(x, axes=None, name=None):
    """The inverse of fftshift.

  Although identical for even-length x,
  the functions differ by one sample for odd-length x.

  @compatibility(numpy)
  Equivalent to numpy.fft.ifftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
  @end_compatibility

  For example:

  ```python
  x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
  x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
  ```

  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple` Axes over which to calculate. Defaults to None,
      which shifts all axes.
    name: An optional name for the operation.

  Returns:
    A `Tensor`, The shifted tensor.
  """
    with _ops.name_scope(name, 'ifftshift') as name:
        x = _ops.convert_to_tensor(x)
        if axes is None:
            axes = tuple(range(x.shape.ndims))
            shift = -(_array_ops.shape(x) // 2)
        elif isinstance(axes, int):
            shift = -(_array_ops.shape(x)[axes] // 2)
        else:
            rank = _array_ops.rank(x)
            axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
            shift = -(_array_ops.gather(_array_ops.shape(x), axes) // 2)
        return manip_ops.roll(x, shift, axes, name)