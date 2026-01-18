import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['image.extract_image_patches', 'extract_image_patches'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'ksizes is deprecated, use sizes instead', 'ksizes')
def extract_image_patches(images, ksizes=None, strides=None, rates=None, padding=None, name=None, sizes=None):
    """Extract patches from images and put them in the "depth" output dimension.

  Args:
    `images`: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`,
      `uint16`, `half`, `uint32`, `uint64`. 4-D Tensor with shape
    `[batch, in_rows, in_cols, depth]`. `ksizes`: A list of `ints` that has
      length `>= 4`. The size of the sliding window for each
    dimension of `images`. `strides`: A list of `ints` that has length `>= 4`.
      1-D of length 4. How far the centers of two consecutive
    patches are in the images. Must be:
    `[1, stride_rows, stride_cols, 1]`. `rates`: A list of `ints`
    that has length `>= 4`. 1-D of length 4. Must be: `[1, rate_rows, rate_cols,
      1]`. This is the input stride, specifying how far two consecutive patch
      samples are in the input. Equivalent to extracting patches with
      `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`,
      followed by subsampling them spatially by a factor of `rates`. This is
      equivalent to `rate` in dilated (a.k.a. Atrous) convolutions.
    `padding`: A `string` from: "SAME", "VALID". The type of padding algorithm
      to use.
    We specify the size-related attributes as:  ``` ksizes = [1, ksize_rows,
      ksize_cols, 1] strides = [1, strides_rows, strides_cols, 1] rates = [1,
      rates_rows, rates_cols, 1]
    name: A name for the operation (optional). ```

  Returns:
    A Tensor. Has the same type as images.
  """
    ksizes = deprecation.deprecated_argument_lookup('sizes', sizes, 'ksizes', ksizes)
    return gen_array_ops.extract_image_patches(images, ksizes, strides, rates, padding, name)