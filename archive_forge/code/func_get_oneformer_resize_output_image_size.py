import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def get_oneformer_resize_output_image_size(image: np.ndarray, size: Union[int, Tuple[int, int], List[int], Tuple[int]], max_size: Optional[int]=None, default_to_square: bool=True, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> tuple:
    """
    Computes the output size given the desired size.

    Args:
        image (`np.ndarray`):
            The input image.
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            The size of the output image.
        max_size (`int`, *optional*):
            The maximum size of the output image.
        default_to_square (`bool`, *optional*, defaults to `True`):
            Whether to default to square if no size is provided.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `Tuple[int, int]`: The output size.
    """
    output_size = get_resize_output_image_size(input_image=image, size=size, default_to_square=default_to_square, max_size=max_size, input_data_format=input_data_format)
    return output_size