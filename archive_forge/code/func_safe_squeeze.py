import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def safe_squeeze(arr: np.ndarray, axis: Optional[int]=None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr