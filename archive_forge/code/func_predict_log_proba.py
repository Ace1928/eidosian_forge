import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
def predict_log_proba(self, X):
    """Return the log of probability estimates.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        log_y_prob : ndarray of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to `log(predict_proba(X))`.
        """
    y_prob = self.predict_proba(X)
    return np.log(y_prob, out=y_prob)