import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def smoothed_state_autocovariance(self, lag=1, t=None, start=None, end=None, extend_kwargs=None):
    """
        Compute state vector autocovariances, conditional on the full dataset

        Computes:

        .. math::

            Cov(\\alpha_t - \\hat \\alpha_t, \\alpha_{t - j} - \\hat \\alpha_{t - j})

        where the `lag` argument gives the value for :math:`j`. Thus when
        the `lag` argument is positive, the autocovariance is between the
        current and previous periods, while if `lag` is negative the
        autocovariance is between the current and future periods.

        Parameters
        ----------
        lag : int, optional
            The number of period to shift when computing the autocovariance.
            Default is 1.
        t : int, optional
            A specific period for which to compute and return the
            autocovariance. Cannot be used in combination with `start` or
            `end`. See the Returns section for details on how this
            parameter affects what is what is returned.
        start : int, optional
            The start of the interval (inclusive) of autocovariances to compute
            and return. Cannot be used in combination with the `t` argument.
            See the Returns section for details on how this parameter affects
            what is what is returned. Default is 0.
        end : int, optional
            The end of the interval (exclusive) autocovariances to compute and
            return. Note that since it is an exclusive endpoint, the returned
            autocovariances do not include the value at this index. Cannot be
            used in combination with the `t` argument. See the Returns section
            for details on how this parameter affects what is what is returned
            and what the default value is.
        extend_kwargs : dict, optional
            Keyword arguments containing updated state space system matrices
            for handling out-of-sample autocovariance computations in
            time-varying state space models.

        Returns
        -------
        acov : ndarray
            Array of autocovariance matrices. If the argument `t` is not
            provided, then it is shaped `(k_states, k_states, n)`, while if `t`
            given then the third axis is dropped and the array is shaped
            `(k_states, k_states)`.

            The output under the default case differs somewhat based on the
            state space model and the sign of the lag. To see how these cases
            differ, denote the output at each time point as Cov(t, t-j). Then:

            - If `lag > 0` (and the model is either time-varying or
              time-invariant), then the returned array is shaped `(*, *, nobs)`
              and each entry [:, :, t] contains Cov(t, t-j). However, the model
              does not have enough information to compute autocovariances in
              the pre-sample period, so that we cannot compute Cov(1, 1-lag),
              Cov(2, 2-lag), ..., Cov(lag, 0). Thus the first `lag` entries
              have all values set to NaN.

            - If the model is time-invariant and `lag < -1` or if `lag` is
              0 or -1, and the model is either time-invariant or time-varying,
              then the returned array is shaped `(*, *, nobs)` and each
              entry [:, :, t] contains Cov(t, t+j). Moreover, all entries are
              available (i.e. there are no NaNs).

            - If the model is time-varying and `lag < -1` and `extend_kwargs`
              is not provided, then the returned array is shaped
              `(*, *, nobs - lag + 1)`.

            - However, if the model is time-varying and `lag < -1`, then
              `extend_kwargs` can be provided with `lag - 1` additional
              matrices so that the returned array is shaped `(*, *, nobs)` as
              usual.

            More generally, the dimension of the last axis will be
            `start - end`.

        Notes
        -----
        This method computes:

        .. math::

            Cov(\\alpha_t - \\hat \\alpha_t, \\alpha_{t - j} - \\hat \\alpha_{t - j})

        where the `lag` argument determines the autocovariance order :math:`j`,
        and `lag` is an integer (positive, zero, or negative). This method
        cannot compute values associated with time points prior to the sample,
        and so it returns a matrix of NaN values for these time points.
        For example, if `start=0` and `lag=2`, then assuming the output is
        assigned to the variable `acov`, we will have `acov[..., 0]` and
        `acov[..., 1]` as matrices filled with NaN values.

        Based only on the "current" results object (i.e. the Kalman smoother
        applied to the sample), there is not enough information to compute
        Cov(t, t+j) for the last `lag - 1` observations of the sample. However,
        the values can be computed for these time points using the transition
        equation of the state space representation, and so for time-invariant
        state space models we do compute these values. For time-varying models,
        this can also be done, but updated state space matrices for the
        out-of-sample time points must be provided via the `extend_kwargs`
        argument.

        See [1]_, Chapter 4.7, for all details about how these autocovariances
        are computed.

        The `t` and `start`/`end` parameters compute and return only the
        requested autocovariances. As a result, using these parameters is
        recommended to reduce the computational burden, particularly if the
        number of observations and/or the dimension of the state vector is
        large.

        References
        ----------
        .. [1] Durbin, James, and Siem Jan Koopman. 2012.
               Time Series Analysis by State Space Methods: Second Edition.
               Oxford University Press.
        """
    cache_key = None
    if extend_kwargs is None or len(extend_kwargs) == 0:
        cache_key = (lag, t, start, end)
    if cache_key is not None and cache_key in self.__smoothed_state_autocovariance:
        return self.__smoothed_state_autocovariance[cache_key]
    forward_autocovariances = False
    if lag < 0:
        lag = -lag
        forward_autocovariances = True
    if t is not None and (start is not None or end is not None):
        raise ValueError('Cannot specify both `t` and `start` or `end`.')
    if t is not None:
        start = t
        end = t + 1
    if start is None:
        start = 0
    if end is None:
        if forward_autocovariances and lag > 1 and (extend_kwargs is None):
            end = self.nobs - lag + 1
        else:
            end = self.nobs
    if extend_kwargs is None:
        extend_kwargs = {}
    if start < 0 or end < 0:
        raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
    if end < start:
        raise ValueError('`end` must be after `start`')
    if lag == 0 and self.smoothed_state_cov is None:
        raise RuntimeError('Cannot return smoothed state covariances if those values have not been computed by Kalman smoothing.')
    if lag == 0 and end <= self.nobs + 1:
        acov = self.smoothed_state_cov
        if end == self.nobs + 1:
            acov = np.concatenate((acov[..., start:], self.predicted_state_cov[..., -1:]), axis=2).T
        else:
            acov = acov.T[start:end]
    elif lag == 1 and self.smoothed_state_autocov is not None and (not forward_autocovariances) and (end <= self.nobs + 1):
        if start == 0:
            nans = np.zeros((self.k_states, self.k_states, lag)) * np.nan
            acov = np.concatenate((nans, self.smoothed_state_autocov[..., :end - 1]), axis=2)
        else:
            acov = self.smoothed_state_autocov[..., start - 1:end - 1]
        acov = acov.transpose(2, 0, 1)
    elif lag == 1 and self.smoothed_state_autocov is not None and forward_autocovariances and (end < self.nobs + 1):
        acov = self.smoothed_state_autocov.T[start:end]
    elif forward_autocovariances:
        acov = self._smoothed_state_autocovariance(lag, start, end, extend_kwargs=extend_kwargs)
    else:
        out = self._smoothed_state_autocovariance(lag, start - lag, end - lag, extend_kwargs=extend_kwargs)
        acov = out.transpose(0, 2, 1)
    if t is not None:
        acov = acov[0]
    else:
        acov = acov.transpose(1, 2, 0)
    if cache_key is not None:
        self.__smoothed_state_autocovariance[cache_key] = acov
    return acov