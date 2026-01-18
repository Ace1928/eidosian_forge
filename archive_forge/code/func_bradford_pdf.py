import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def bradford_pdf(x, c):
    if 0 <= x <= 1:
        return 1.0 / (1.0 + c * x)
    return 0.0