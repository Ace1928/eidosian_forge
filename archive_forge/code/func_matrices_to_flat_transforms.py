from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.gen_image_ops import *
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops.image_ops_impl import _Check3DImage
from tensorflow.python.ops.image_ops_impl import _ImageDimensions
def matrices_to_flat_transforms(transform_matrices):
    """Converts affine matrices to `tf.contrib.image` projective transforms.

  Note that we expect matrices that map output coordinates to input coordinates.
  To convert forward transformation matrices, call `tf.linalg.inv` on the
  matrices and use the result here.

  Args:
    transform_matrices: One or more affine transformation matrices, for the
      reverse transformation in homogeneous coordinates. Shape `(3, 3)` or `(N,
      3, 3)`.

  Returns:
    2D tensor of flat transforms with shape `(N, 8)`, which may be passed into
      `tf.contrib.image.transform`.

  Raises:
    ValueError: If `transform_matrices` have an invalid shape.
  """
    with ops.name_scope('matrices_to_flat_transforms'):
        transform_matrices = ops.convert_to_tensor(transform_matrices, name='transform_matrices')
        if transform_matrices.shape.ndims not in (2, 3):
            raise ValueError('Matrices should be 2D or 3D, got: %s' % transform_matrices)
        transforms = array_ops.reshape(transform_matrices, constant_op.constant([-1, 9]))
        transforms /= transforms[:, 8:9]
        return transforms[:, :8]