from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
    """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    default_to_square = True
    if 'shortest_edge' in size:
        size = size['shortest_edge']
        default_to_square = False
    elif 'height' in size and 'width' in size:
        size = (size['height'], size['width'])
    else:
        raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
    output_size = get_resize_output_image_size(image, size=size, default_to_square=default_to_square, input_data_format=input_data_format)
    return resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)