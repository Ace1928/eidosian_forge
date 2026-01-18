import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def geninvgauss_mode(p, b):
    if p > 1:
        return (math.sqrt((1 - p) ** 2 + b ** 2) - (1 - p)) / b
    return b / (math.sqrt((1 - p) ** 2 + b ** 2) + (1 - p))