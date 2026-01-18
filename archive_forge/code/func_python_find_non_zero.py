from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available
def python_find_non_zero(self, image: np.array):
    """This is a reimplementation of a findNonZero function equivalent to cv2."""
    non_zero_indices = np.column_stack(np.nonzero(image))
    idxvec = non_zero_indices[:, [1, 0]]
    idxvec = idxvec.reshape(-1, 1, 2)
    return idxvec