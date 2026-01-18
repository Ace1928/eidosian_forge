import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def geninvgauss_pdf(x, p, b):
    m = geninvgauss_mode(p, b)
    lfm = (p - 1) * math.log(m) - 0.5 * b * (m + 1 / m)
    if x > 0:
        return math.exp((p - 1) * math.log(x) - 0.5 * b * (x + 1 / x) - lfm)
    return 0.0