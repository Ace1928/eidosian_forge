import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def asymptotic(dim, df):
    return (dim * norm._entropy() + dim / df - dim * (dim - 2) * df ** (-2.0) / 4 + dim ** 2 * (dim - 2) * df ** (-3.0) / 6 + dim * (-3 * dim ** 3 + 8 * dim ** 2 - 8) * df ** (-4.0) / 24 + dim ** 2 * (3 * dim ** 3 - 10 * dim ** 2 + 16) * df ** (-5.0) / 30 + shape_term)[()]