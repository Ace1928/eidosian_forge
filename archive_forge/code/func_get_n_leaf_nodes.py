import numpy as np
from ._predictor import (
from .common import PREDICTOR_RECORD_DTYPE, Y_DTYPE
def get_n_leaf_nodes(self):
    """Return number of leaves."""
    return int(self.nodes['is_leaf'].sum())