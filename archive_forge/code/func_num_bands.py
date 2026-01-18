from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
@property
def num_bands(self):
    """Number of bands."""
    return len(self.center_frequencies)