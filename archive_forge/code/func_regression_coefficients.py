from warnings import warn
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
@property
def regression_coefficients(self):
    """
        Estimates of unobserved regression coefficients

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
    out = None
    spec = self.specification
    if spec.regression:
        if spec.mle_regression:
            import warnings
            warnings.warn('Regression coefficients estimated via maximum likelihood. Estimated coefficients are available in the parameters list, not as part of the state vector.', OutputWarning)
        else:
            offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'] + self._k_states_by_type['freq_seasonal'] + self._k_states_by_type['cycle'] + spec.ar_order)
            start = offset
            end = offset + spec.k_exog
            out = Bunch(filtered=self.filtered_state[start:end], filtered_cov=self.filtered_state_cov[start:end, start:end], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[start:end]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[start:end, start:end]
    return out