import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
@deprecated('The function `delayed` has been moved from `sklearn.utils.fixes` to `sklearn.utils.parallel`. This import path will be removed in 1.5.')
def delayed(function):
    from sklearn.utils.parallel import delayed
    return delayed(function)