from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import flip_channel_order, get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
def flip_channel_order(self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """
        Flip the color channels from RGB to BGR or vice versa.

        Args:
            image (`np.ndarray`):
                The image, represented as a numpy array.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    return flip_channel_order(image, data_format=data_format, input_data_format=input_data_format)