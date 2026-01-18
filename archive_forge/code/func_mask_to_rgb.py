from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, logging, requires_backends
def mask_to_rgb(self, image: np.ndarray, palette: Optional[List[Tuple[int, int]]]=None, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """Convert a mask to RGB format.

        Args:
            image (`np.ndarray`):
                Mask to convert to RGB format. If the mask is already in RGB format, it will be passed through.
            palette (`List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Palette to use to convert the mask to RGB format. If unset, the mask is duplicated across the channel
                dimension.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The mask in RGB format.
        """
    return mask_to_rgb(image, palette=palette, data_format=data_format, input_data_format=input_data_format)