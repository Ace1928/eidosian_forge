import math
from typing import Optional
import numpy as np
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def num_quantizers(self) -> int:
    return int(1000 * self.target_bandwidths[-1] // (self.frame_rate * 10))