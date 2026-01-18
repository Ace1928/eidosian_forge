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
@tf_export('image.ssim_multiscale')
@dispatch.add_dispatch_support
def ssim_multiscale(img1, img2, max_val, power_factors=_MSSSIM_WEIGHTS, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """Computes the MS-SSIM between img1 and img2.

  This function assumes that `img1` and `img2` are image batches, i.e. the last
  three dimensions are [height, width, channels].

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)

  Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
  structural similarity for image quality assessment." Signals, Systems and
  Computers, 2004.

  Args:
    img1: First image batch with only Positive Pixel Values.
    img2: Second image batch with only Positive Pixel Values. Must have the
    same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the original paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).

  Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
    with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)
        imgs = [img1, img2]
        shapes = [shape1, shape2]
        heads = [s[:-3] for s in shapes]
        tails = [s[-3:] for s in shapes]
        divisor = [1, 2, 2, 1]
        divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

        def do_pad(images, remainder):
            padding = array_ops.expand_dims(remainder, -1)
            padding = array_ops.pad(padding, [[1, 0], [1, 0]])
            return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]
        mcs = []
        for k in range(len(power_factors)):
            with ops.name_scope(None, 'Scale%d' % k, imgs):
                if k > 0:
                    flat_imgs = [array_ops.reshape(x, array_ops.concat([[-1], t], 0)) for x, t in zip(imgs, tails)]
                    remainder = tails[0] % divisor_tensor
                    need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
                    padded = tf_cond.cond(need_padding, lambda: do_pad(flat_imgs, remainder), lambda: flat_imgs)
                    downscaled = [nn_ops.avg_pool(x, ksize=divisor, strides=divisor, padding='VALID') for x in padded]
                    tails = [x[1:] for x in array_ops.shape_n(downscaled)]
                    imgs = [array_ops.reshape(x, array_ops.concat([h, t], 0)) for x, h, t in zip(downscaled, heads, tails)]
                ssim_per_channel, cs = _ssim_per_channel(*imgs, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
                mcs.append(nn_ops.relu(cs))
        mcs.pop()
        mcs_and_ssim = array_ops_stack.stack(mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
        ms_ssim = math_ops.reduce_prod(math_ops.pow(mcs_and_ssim, power_factors), [-1])
        return math_ops.reduce_mean(ms_ssim, [-1])