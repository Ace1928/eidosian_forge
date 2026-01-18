import numpy as np
from ._predictor import (
from .common import PREDICTOR_RECORD_DTYPE, Y_DTYPE
def compute_partial_dependence(self, grid, target_features, out):
    """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
    _compute_partial_dependence(self.nodes, grid, target_features, out)