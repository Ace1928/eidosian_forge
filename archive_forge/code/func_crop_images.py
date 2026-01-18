from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
@keras_export('keras.ops.image.crop_images')
def crop_images(images, top_cropping=None, left_cropping=None, target_height=None, target_width=None, bottom_cropping=None, right_cropping=None):
    """Crop `images` to a specified `height` and `width`.

    Args:
        images: 4-D batch of images of shape `(batch, height, width, channels)`
             or 3-D single image of shape `(height, width, channels)`.
        top_cropping: Number of columns to crop from the top.
        bottom_cropping: Number of columns to crop from the bottom.
        left_cropping: Number of columns to crop from the left.
        right_cropping: Number of columns to crop from the right.
        target_height: Height of the output images.
        target_width: Width of the output images.

    Returns:
        If `images` were 4D, a 4D float Tensor of shape
            `(batch, target_height, target_width, channels)`
        If `images` were 3D, a 3D float Tensor of shape
            `(target_height, target_width, channels)`

    Example:

    >>> images = np.reshape(np.arange(1, 28, dtype="float32"), [3, 3, 3])
    >>> images[:,:,0] # print the first channel of the images
    array([[ 1.,  4.,  7.],
           [10., 13., 16.],
           [19., 22., 25.]], dtype=float32)
    >>> cropped_images = keras.image.crop_images(images, 0, 0, 2, 2)
    >>> cropped_images[:,:,0] # print the first channel of the cropped images
    array([[ 1.,  4.],
           [10., 13.]], dtype=float32)"""
    if any_symbolic_tensors((images,)):
        return CropImages(top_cropping, bottom_cropping, left_cropping, right_cropping, target_height, target_width).symbolic_call(images)
    return _crop_images(images, top_cropping, bottom_cropping, left_cropping, right_cropping, target_height, target_width)