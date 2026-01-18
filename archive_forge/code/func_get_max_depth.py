import numpy as np
from ._predictor import (
from .common import PREDICTOR_RECORD_DTYPE, Y_DTYPE
def get_max_depth(self):
    """Return maximum depth among all leaves."""
    return int(self.nodes['depth'].max())