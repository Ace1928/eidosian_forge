import math
from typing import Optional
import numpy as np
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def frame_rate(self) -> int:
    hop_length = np.prod(self.upsampling_ratios)
    return math.ceil(self.sampling_rate / hop_length)