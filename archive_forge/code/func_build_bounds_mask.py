import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
def build_bounds_mask(vector):
    """
    Creates masks for converting from the full vector of bounds that
    may contain -np.inf or np.inf to a vector of bounds that are finite
    only.
    """
    return build_compression_mask_for_finite_values(vector)