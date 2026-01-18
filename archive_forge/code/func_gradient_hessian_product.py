import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def gradient_hessian_product(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1):
    """Computes gradient and hessp (hessian product function) w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.

        hessp : callable
            Function that takes in a vector input of shape of gradient and
            and returns matrix-vector product with hessian.
        """
    (n_samples, n_features), n_classes = (X.shape, self.base_loss.n_classes)
    n_dof = n_features + int(self.fit_intercept)
    weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
    sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
    if not self.base_loss.is_multiclass:
        grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
        grad_pointwise /= sw_sum
        hess_pointwise /= sw_sum
        grad = np.empty_like(coef, dtype=weights.dtype)
        grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
        if self.fit_intercept:
            grad[-1] = grad_pointwise.sum()
        hessian_sum = hess_pointwise.sum()
        if sparse.issparse(X):
            hX = sparse.dia_matrix((hess_pointwise, 0), shape=(n_samples, n_samples)) @ X
        else:
            hX = hess_pointwise[:, np.newaxis] * X
        if self.fit_intercept:
            hX_sum = np.squeeze(np.asarray(hX.sum(axis=0)))
            hX_sum = np.atleast_1d(hX_sum)

        def hessp(s):
            ret = np.empty_like(s)
            if sparse.issparse(X):
                ret[:n_features] = X.T @ (hX @ s[:n_features])
            else:
                ret[:n_features] = np.linalg.multi_dot([X.T, hX, s[:n_features]])
            ret[:n_features] += l2_reg_strength * s[:n_features]
            if self.fit_intercept:
                ret[:n_features] += s[-1] * hX_sum
                ret[-1] = hX_sum @ s[:n_features] + hessian_sum * s[-1]
            return ret
    else:
        grad_pointwise, proba = self.base_loss.gradient_proba(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
        grad_pointwise /= sw_sum
        grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
        grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
        if self.fit_intercept:
            grad[:, -1] = grad_pointwise.sum(axis=0)

        def hessp(s):
            s = s.reshape((n_classes, -1), order='F')
            if self.fit_intercept:
                s_intercept = s[:, -1]
                s = s[:, :-1]
            else:
                s_intercept = 0
            tmp = X @ s.T + s_intercept
            tmp += (-proba * tmp).sum(axis=1)[:, np.newaxis]
            tmp *= proba
            if sample_weight is not None:
                tmp *= sample_weight[:, np.newaxis]
            hess_prod = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
            hess_prod[:, :n_features] = tmp.T @ X / sw_sum + l2_reg_strength * s
            if self.fit_intercept:
                hess_prod[:, -1] = tmp.sum(axis=0) / sw_sum
            if coef.ndim == 1:
                return hess_prod.ravel(order='F')
            else:
                return hess_prod
        if coef.ndim == 1:
            return (grad.ravel(order='F'), hessp)
    return (grad, hessp)