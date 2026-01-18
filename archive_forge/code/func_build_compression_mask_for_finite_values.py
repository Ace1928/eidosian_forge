import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
def build_compression_mask_for_finite_values(vector):
    """
    Creates masks for converting from the full vector of
    values to the vector that contains only the finite values. This is
    typically used to convert a vector of bounds (that may contain np.inf
    and -np.inf) to only the bounds that are finite.
    """
    full_finite_mask = np.isfinite(vector)
    return full_finite_mask