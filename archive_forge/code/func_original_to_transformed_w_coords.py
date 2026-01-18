import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)