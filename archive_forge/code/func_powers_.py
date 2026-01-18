import collections
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from numbers import Integral
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import comb
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
from ._csr_polynomial_expansion import (
@property
def powers_(self):
    """Exponent for each of the inputs in the output."""
    check_is_fitted(self)
    combinations = self._combinations(n_features=self.n_features_in_, min_degree=self._min_degree, max_degree=self._max_degree, interaction_only=self.interaction_only, include_bias=self.include_bias)
    return np.vstack([np.bincount(c, minlength=self.n_features_in_) for c in combinations])