from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def get_smoothed_decomposition(self, decomposition_of='smoothed_state', state_index=None, original_scale=True):
    """
        Decompose smoothed output into contributions from observations

        Parameters
        ----------
        decomposition_of : {"smoothed_state", "smoothed_signal"}
            The object to perform a decomposition of. If it is set to
            "smoothed_state", then the elements of the smoothed state vector
            are decomposed into the contributions of each observation. If it
            is set to "smoothed_signal", then the predictions of the
            observation vector based on the smoothed state vector are
            decomposed. Default is "smoothed_state".
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the decomposition of the "smoothed_signal". For
            example, if `state_index=[0, 1]` is passed, then only the
            contributions of observed variables to the smoothed signal arising
            from the first two states will be returned. Note that if not all
            states are used, the contributions will not sum to the smoothed
            signal. Default is to use all states.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return simulations in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.

        Returns
        -------
        data_contributions : pd.DataFrame
            Contributions of observations to the decomposed object. If the
            smoothed state is being decomposed, then `data_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `variable_from x date_from`. If the
            smoothed signal is being decomposed, then `data_contributions` is
            shaped `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `variable_from x date_from`.
        obs_intercept_contributions : pd.DataFrame
            Contributions of the observation intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `obs_intercept_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `obs_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `obs_intercept_contributions` is shaped
            `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `obs_intercept_from x date_from`.
        state_intercept_contributions : pd.DataFrame
            Contributions of the state intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `state_intercept_contributions` is
            shaped `(k_states x nobs, k_states x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `state_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `state_intercept_contributions` is shaped
            `(k_endog x nobs, k_states x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `state_intercept_from x date_from`.
        prior_contributions : pd.DataFrame
            Contributions of the prior to the decomposed object. If the
            smoothed state is being decomposed, then `prior_contributions` is
            shaped `(nobs x k_states, k_states)`, with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and columns
            corresponding to elements of the prior mean (aka "initial state").
            If the smoothed signal is being decomposed, then
            `prior_contributions` is shaped `(nobs x k_endog, k_states)`,
            with a `pd.MultiIndex` index corresponding to
            `variable_to x date_to` and columns corresponding to elements of
            the prior mean.

        Notes
        -----
        Denote the smoothed state at time :math:`t` by :math:`\\alpha_t`. Then
        the smoothed signal is :math:`Z_t \\alpha_t`, where :math:`Z_t` is the
        design matrix operative at time :math:`t`.
        """
    if self.model.standardize and original_scale:
        cache_obs_intercept = self.model['obs_intercept']
        self.model['obs_intercept'] = self.model._endog_mean
    data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions = super().get_smoothed_decomposition(decomposition_of=decomposition_of, state_index=state_index)
    if self.model.standardize and original_scale:
        self.model['obs_intercept'] = cache_obs_intercept
    if decomposition_of == 'smoothed_signal' and self.model.standardize and original_scale:
        endog_std = self.model._endog_std
        data_contributions = data_contributions.multiply(endog_std, axis=0, level=0)
        obs_intercept_contributions = obs_intercept_contributions.multiply(endog_std, axis=0, level=0)
        state_intercept_contributions = state_intercept_contributions.multiply(endog_std, axis=0, level=0)
        prior_contributions = prior_contributions.multiply(endog_std, axis=0, level=0)
    return (data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions)