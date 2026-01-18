import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def map_pixels(self, image: np.ndarray) -> np.ndarray:
    return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS