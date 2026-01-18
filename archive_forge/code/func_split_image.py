from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def split_image(self, image: np.ndarray, input_data_format: Optional[Union[str, ChannelDimension]]=None):
    """
        Split an image into 4 equal sub-images, and the concatenate that sequence with the original image.
        That means that a single image becomes a sequence of 5 images.
        This is a "trick" to spend more compute on each image with no changes in the vision encoder.

        Args:
            image (`np.ndarray`):
                Images to split.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    height, width = get_image_size(image, input_data_format)
    mid_width = width // 2
    mid_height = height // 2
    return [self._crop(image, 0, 0, mid_width, mid_height, input_data_format), self._crop(image, mid_width, 0, width, mid_height, input_data_format), self._crop(image, 0, mid_height, mid_width, height, input_data_format), self._crop(image, mid_width, mid_height, width, height, input_data_format), image]