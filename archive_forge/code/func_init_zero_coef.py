import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def init_zero_coef(self, X, dtype=None):
    """Allocate coef of correct shape with zeros.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        dtype : data-type, default=None
            Overrides the data type of coef. With dtype=None, coef will have the same
            dtype as X.

        Returns
        -------
        coef : ndarray of shape (n_dof,) or (n_classes, n_dof)
            Coefficients of a linear model.
        """
    n_features = X.shape[1]
    n_classes = self.base_loss.n_classes
    if self.fit_intercept:
        n_dof = n_features + 1
    else:
        n_dof = n_features
    if self.base_loss.is_multiclass:
        coef = np.zeros_like(X, shape=(n_classes, n_dof), dtype=dtype, order='F')
    else:
        coef = np.zeros_like(X, shape=n_dof, dtype=dtype)
    return coef