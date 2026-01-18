import numbers
from functools import partial
import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike
from ._sparsetools import csr_hstack
from ._csr import csr_matrix
from ._csc import csc_matrix
from ._bsr import bsr_matrix
from ._coo import coo_matrix
from ._dia import dia_matrix
from ._base import issparse
def data_rvs(n):
    return random_state.uniform(size=n) + random_state.uniform(size=n) * 1j