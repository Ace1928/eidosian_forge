import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
def reduce_label(self, label: ImageInput) -> np.ndarray:
    label = to_numpy_array(label)
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label