from typing import Dict, Optional, Union
import numpy as np
from ... import is_vision_available
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, logging, requires_backends
def is_grayscale(image: ImageInput, input_data_format: Optional[Union[str, ChannelDimension]]=None):
    if input_data_format == ChannelDimension.FIRST:
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
    elif input_data_format == ChannelDimension.LAST:
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])