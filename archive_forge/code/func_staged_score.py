import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
def staged_score(self, X, y, sample_weight=None):
    """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
    X = self._check_X(X)
    for y_pred in self.staged_predict(X):
        if is_classifier(self):
            yield accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            yield r2_score(y, y_pred, sample_weight=sample_weight)