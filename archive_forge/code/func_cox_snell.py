import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def cox_snell(x):
    return special.betainc(2 / 3.0, 2 / 3.0, x) * special.beta(2 / 3.0, 2 / 3.0)