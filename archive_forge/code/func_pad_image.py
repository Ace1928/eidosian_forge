from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available
def pad_image(self, image: np.ndarray, size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """
        Pad the image to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
    output_height, output_width = (size['height'], size['width'])
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    delta_width = output_width - input_width
    delta_height = output_height - input_height
    pad_top = delta_height // 2
    pad_left = delta_width // 2
    pad_bottom = delta_height - pad_top
    pad_right = delta_width - pad_left
    padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    return pad(image, padding, data_format=data_format, input_data_format=input_data_format)