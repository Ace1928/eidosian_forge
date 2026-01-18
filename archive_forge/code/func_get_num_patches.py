import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def get_num_patches(self, image_height: int, image_width: int, patch_size: Dict[str, int]=None) -> int:
    """
        Calculate number of patches required to encode an image.

        Args:
            image_height (`int`):
                Height of the image.
            image_width (`int`):
                Width of the image.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
    patch_size = patch_size if patch_size is not None else self.patch_size
    patch_height, patch_width = (self.patch_size['height'], self.patch_size['width'])
    if image_height % patch_height != 0:
        raise ValueError(f'image_height={image_height!r} must be divisible by {patch_height}')
    if image_width % patch_width != 0:
        raise ValueError(f'image_width={image_width!r} must be divisible by {patch_width}')
    num_patches_per_dim_h = image_height // patch_height
    num_patches_per_dim_w = image_width // patch_width
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches