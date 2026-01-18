import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension
from .utils import (
def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (int(original_width * scale), int(original_height * scale))
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = width * height - effective_resolution
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)
    return best_fit