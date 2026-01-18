import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def get_image_patches(self, image: np.array, grid_pinpoints, size: tuple, patch_size: int, resample: PILImageResampling, data_format: ChannelDimension, input_data_format: ChannelDimension) -> List[np.array]:
    """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """
    if not isinstance(grid_pinpoints, list):
        raise ValueError('grid_pinpoints must be a list of possible resolutions.')
    possible_resolutions = grid_pinpoints
    image_size = get_image_size(image, channel_dim=input_data_format)
    best_resolution = select_best_resolution(image_size, possible_resolutions)
    resized_image = self._resize_for_patching(image, best_resolution, resample=resample, input_data_format=input_data_format)
    padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)
    patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)
    patches = [to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format) for patch in patches]
    resized_original_image = resize(image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format)
    image_patches = [resized_original_image] + patches
    return image_patches