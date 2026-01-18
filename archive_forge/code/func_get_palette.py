from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, logging, requires_backends
def get_palette(self, num_labels: int) -> List[Tuple[int, int]]:
    """Build a palette to map the prompt mask from a single channel to a 3 channel RGB.

        Args:
            num_labels (`int`):
                Number of classes in the segmentation task (excluding the background).

        Returns:
            `List[Tuple[int, int]]`: Palette to map the prompt mask from a single channel to a 3 channel RGB.
        """
    return build_palette(num_labels)