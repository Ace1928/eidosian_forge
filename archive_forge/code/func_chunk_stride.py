import math
from typing import Optional
import numpy as np
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def chunk_stride(self) -> Optional[int]:
    if self.chunk_length_s is None or self.overlap is None:
        return None
    else:
        return max(1, int((1.0 - self.overlap) * self.chunk_length))