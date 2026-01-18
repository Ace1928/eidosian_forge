from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(image_ops.resize_images)
def resize_images_v1(images: ragged_tensor.RaggedTensor, size, method=image_ops.ResizeMethodV1.BILINEAR, align_corners=False, preserve_aspect_ratio=False, name=None):
    """RaggedTensor dispatcher for tf.image.resize (tf-v1)."""
    with ops.name_scope(name, 'RaggedResizeImages', [images, size]):
        return _resize_images(image_ops.resize_images, images, size, method=method, preserve_aspect_ratio=preserve_aspect_ratio, align_corners=align_corners)