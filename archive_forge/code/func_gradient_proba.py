import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
def gradient_proba(self, y_true, raw_prediction, sample_weight=None, gradient_out=None, proba_out=None, n_threads=1):
    """Compute gradient and class probabilities fow raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
    if gradient_out is None:
        if proba_out is None:
            gradient_out = np.empty_like(raw_prediction)
            proba_out = np.empty_like(raw_prediction)
        else:
            gradient_out = np.empty_like(proba_out)
    elif proba_out is None:
        proba_out = np.empty_like(gradient_out)
    self.closs.gradient_proba(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, gradient_out=gradient_out, proba_out=proba_out, n_threads=n_threads)
    return (gradient_out, proba_out)