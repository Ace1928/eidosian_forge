import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ncf_sf(x, dfn, dfd, nc):
    return 1 - special.ncfdtr(dfn, dfd, nc, x)