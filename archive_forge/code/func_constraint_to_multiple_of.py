import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def constraint_to_multiple_of(val, multiple, min_val=0, max_val=None):
    x = round(val / multiple) * multiple
    if max_val is not None and x > max_val:
        x = math.floor(val / multiple) * multiple
    if x < min_val:
        x = math.ceil(val / multiple) * multiple
    return x