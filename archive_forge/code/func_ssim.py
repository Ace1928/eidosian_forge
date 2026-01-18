import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('image.ssim')
@dispatch.add_dispatch_support
def ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=False):
    """Computes SSIM index between img1 and img2.

  This function is based on the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  The image sizes must be at least 11x11 because of the filter size.

  Example:

  ```python
      # Read images (of size 255 x 255) from file.
      im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
      im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
      tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
      tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
      # Add an outer batch for each image.
      im1 = tf.expand_dims(im1, axis=0)
      im2 = tf.expand_dims(im2, axis=0)
      # Compute SSIM over tf.uint8 Tensors.
      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)

      # Compute SSIM over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
      # ssim1 and ssim2 both have type tf.float32 and are almost equal.
  ```

  Args:
    img1: First image batch. 4-D Tensor of shape `[batch, height, width,
      channels]` with only Positive Pixel Values.
    img2: Second image batch. 4-D Tensor of shape `[batch, height, width,
      channels]` with only Positive Pixel Values.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
    return_index_map: If True returns local SSIM map instead of the global mean.

  Returns:
    A tensor containing an SSIM value for each image in batch or a tensor
    containing an SSIM value for each pixel for each image in batch if
    return_index_map is True. Returned SSIM values are in range (-1, 1], when
    pixel values are non-negative. Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]) or broadcast(img1.shape[:-1],
    img2.shape[:-1]).
  """
    with ops.name_scope(None, 'SSIM', [img1, img2]):
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        _, _, checks = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)
        ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2, return_index_map)
        return math_ops.reduce_mean(ssim_per_channel, [-1])