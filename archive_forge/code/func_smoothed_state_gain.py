import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def smoothed_state_gain(self, updates_ix, t=None, start=None, end=None, extend_kwargs=None):
    """
        Cov(\\tilde \\alpha_{t}, I) Var(I, I)^{-1}

        where I is a vector of forecast errors associated with
        `update_indices`.

        Parameters
        ----------
        updates_ix : list
            List of indices `(t, i)`, where `t` denotes a zero-indexed time
            location and `i` denotes a zero-indexed endog variable.
        """
    if t is not None and (start is not None or end is not None):
        raise ValueError('Cannot specify both `t` and `start` or `end`.')
    if t is not None:
        start = t
        end = t + 1
    if start is None:
        start = self.nobs - 1
    if end is None:
        end = self.nobs
    if extend_kwargs is None:
        extend_kwargs = {}
    if start < 0 or end < 0:
        raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
    if end <= start:
        raise ValueError('`end` must be after `start`')
    n_periods = end - start
    n_updates = len(updates_ix)

    def get_mat(which, t):
        mat = getattr(self, which)
        if mat.shape[-1] > 1:
            if t < self.nobs:
                out = mat[..., t]
            else:
                if which not in extend_kwargs or extend_kwargs[which].shape[-1] <= t - self.nobs:
                    raise ValueError(f'Model has time-varying {which} matrix, so an updated time-varying matrix for the extension period is required.')
                out = extend_kwargs[which][..., t - self.nobs]
        else:
            out = mat[..., 0]
        return out

    def get_cov_state_revision(t):
        tmp1 = np.zeros((self.k_states, n_updates))
        for i in range(n_updates):
            t_i, k_i = updates_ix[i]
            acov = self.smoothed_state_autocovariance(lag=t - t_i, t=t, extend_kwargs=extend_kwargs)
            Z_i = get_mat('design', t_i)
            tmp1[:, i:i + 1] = acov @ Z_i[k_i:k_i + 1].T
        return tmp1
    tmp1 = np.zeros((n_periods, self.k_states, n_updates))
    for s in range(start, end):
        tmp1[s - start] = get_cov_state_revision(s)
    tmp2 = np.zeros((n_updates, n_updates))
    for i in range(n_updates):
        t_i, k_i = updates_ix[i]
        for j in range(i + 1):
            t_j, k_j = updates_ix[j]
            Z_i = get_mat('design', t_i)
            Z_j = get_mat('design', t_j)
            acov = self.smoothed_state_autocovariance(lag=t_i - t_j, t=t_i, extend_kwargs=extend_kwargs)
            tmp2[i, j] = tmp2[j, i] = np.squeeze(Z_i[k_i:k_i + 1] @ acov @ Z_j[k_j:k_j + 1].T)
            if t_i == t_j:
                H = get_mat('obs_cov', t_i)
                if i == j:
                    tmp2[i, j] += H[k_i, k_j]
                else:
                    tmp2[i, j] += H[k_i, k_j]
                    tmp2[j, i] += H[k_i, k_j]
    gain = tmp1 @ np.linalg.inv(tmp2)
    if t is not None:
        gain = gain[0]
    return gain