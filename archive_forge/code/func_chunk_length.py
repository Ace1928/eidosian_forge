import math
from typing import Optional
import numpy as np
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def chunk_length(self) -> Optional[int]:
    if self.chunk_length_s is None:
        return None
    else:
        return int(self.chunk_length_s * self.sampling_rate)