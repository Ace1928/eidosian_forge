import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def chi_pdf(x, a):
    if x > 0:
        return math.exp((a - 1) * math.log(x) - 0.5 * (x * x) - (a / 2 - 1) * math.log(2) - math.lgamma(0.5 * a))
    else:
        return 0 if a >= 1 else np.inf