import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
@property
def coef_(self):
    """Weights assigned to the features when `kernel="linear"`.

        Returns
        -------
        ndarray of shape (n_features, n_classes)
        """
    if self.kernel != 'linear':
        raise AttributeError('coef_ is only available when using a linear kernel')
    coef = self._get_coef()
    if sp.issparse(coef):
        coef.data.flags.writeable = False
    else:
        coef.flags.writeable = False
    return coef