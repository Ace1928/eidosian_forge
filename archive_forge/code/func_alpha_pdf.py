import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def alpha_pdf(x, a):
    if x > 0:
        return math.exp(-2.0 * math.log(x) - 0.5 * (a - 1.0 / x) ** 2)
    return 0.0