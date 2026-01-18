import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def invgauss_mode(mu):
    return 1.0 / (math.sqrt(1.5 * 1.5 + 1 / (mu * mu)) + 1.5)