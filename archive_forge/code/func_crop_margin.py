from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available
def crop_margin(self, image: np.array, gray_threshold: int=200, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.array:
    """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`np.array`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`)
                Value below which pixels are considered to be gray.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image. If unset, will use the inferred format from the
                input.
            input_data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the input image. If unset, will use the inferred format from the input.
        """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    image = to_pil_image(image, input_data_format=input_data_format)
    data = np.array(image.convert('L')).astype(np.uint8)
    max_val = data.max()
    min_val = data.min()
    if max_val == min_val:
        image = np.array(image)
        image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        return image
    data = (data - min_val) / (max_val - min_val) * 255
    gray = data < gray_threshold
    coords = self.python_find_non_zero(gray)
    x_min, y_min, width, height = self.python_bounding_rect(coords)
    image = image.crop((x_min, y_min, x_min + width, y_min + height))
    image = np.array(image).astype(np.uint8)
    image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image