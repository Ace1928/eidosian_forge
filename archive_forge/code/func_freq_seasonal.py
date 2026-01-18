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
def freq_seasonal(self):
    """
        Estimates of unobserved frequency domain seasonal component(s)

        Returns
        -------
        out: list of Bunch instances
            Each item has the following attributes:

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
    out = []
    spec = self.specification
    if spec.freq_seasonal:
        previous_states_offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'])
        previous_f_seas_offset = 0
        for ix, h in enumerate(spec.freq_seasonal_harmonics):
            offset = previous_states_offset + previous_f_seas_offset
            period = spec.freq_seasonal_periods[ix]
            states_in_sum = np.arange(0, 2 * h, 2)
            filtered_state = np.sum([self.filtered_state[offset + j] for j in states_in_sum], axis=0)
            filtered_cov = np.sum([self.filtered_state_cov[offset + j, offset + j] for j in states_in_sum], axis=0)
            item = Bunch(filtered=filtered_state, filtered_cov=filtered_cov, smoothed=None, smoothed_cov=None, offset=offset, pretty_name='seasonal {p}({h})'.format(p=repr(period), h=repr(h)))
            if self.smoothed_state is not None:
                item.smoothed = np.sum([self.smoothed_state[offset + j] for j in states_in_sum], axis=0)
            if self.smoothed_state_cov is not None:
                item.smoothed_cov = np.sum([self.smoothed_state_cov[offset + j, offset + j] for j in states_in_sum], axis=0)
            out.append(item)
            previous_f_seas_offset += 2 * h
    return out