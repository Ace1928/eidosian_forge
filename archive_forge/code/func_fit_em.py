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
def fit_em(self, start_params=None, transformed=True, cov_type='none', cov_kwds=None, maxiter=500, tolerance=1e-06, disp=False, em_initialization=True, mstep_method=None, full_output=True, return_params=False, low_memory=False, llf_decrease_action='revert', llf_decrease_tolerance=0.0001):
    """
        Fits the model by maximum likelihood via the EM algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is to use `DynamicFactorMQ.start_params`.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'none', since computing this matrix can be very slow
            when there are a large number of parameters.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        maxiter : int, optional
            The maximum number of EM iterations to perform.
        tolerance : float, optional
            Parameter governing convergence of the EM algorithm. The
            `tolerance` is the minimum relative increase in the likelihood
            for which convergence will be declared. A smaller value for the
            `tolerance` will typically yield more precise parameter estimates,
            but will typically require more EM iterations. Default is 1e-6.
        disp : int or bool, optional
            Controls printing of EM iteration progress. If an integer, progress
            is printed at every `disp` iterations. A value of True is
            interpreted as the value of 1. Default is False (nothing will be
            printed).
        em_initialization : bool, optional
            Whether or not to also update the Kalman filter initialization
            using the EM algorithm. Default is True.
        mstep_method : {None, 'missing', 'nonmissing'}, optional
            The EM algorithm maximization step. If there are no NaN values
            in the dataset, this can be set to "nonmissing" (which is slightly
            faster) or "missing", otherwise it must be "missing". Default is
            "nonmissing" if there are no NaN values or "missing" if there are.
        full_output : bool, optional
            Set to True to have all available output from EM iterations in
            the Results object's mle_retvals attribute.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        low_memory : bool, optional
            This option cannot be used with the EM algorithm and will raise an
            error if set to True. Default is False.
        llf_decrease_action : {'ignore', 'warn', 'revert'}, optional
            Action to take if the log-likelihood decreases in an EM iteration.
            'ignore' continues the iterations, 'warn' issues a warning but
            continues the iterations, while 'revert' ends the iterations and
            returns the result from the last good iteration. Default is 'warn'.
        llf_decrease_tolerance : float, optional
            Minimum size of the log-likelihood decrease required to trigger a
            warning or to end the EM iterations. Setting this value slightly
            larger than zero allows small decreases in the log-likelihood that
            may be caused by numerical issues. If set to zero, then any
            decrease will trigger the `llf_decrease_action`. Default is 1e-4.

        Returns
        -------
        DynamicFactorMQResults

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEModel.fit
        statsmodels.tsa.statespace.mlemodel.MLEResults
        """
    if self._has_fixed_params:
        raise NotImplementedError('Cannot fit using the EM algorithm while holding some parameters fixed.')
    if low_memory:
        raise ValueError('Cannot fit using the EM algorithm when using low_memory option.')
    if start_params is None:
        start_params = self.start_params
        transformed = True
    else:
        start_params = np.array(start_params, ndmin=1)
    if not transformed:
        start_params = self.transform_params(start_params)
    llf_decrease_action = string_like(llf_decrease_action, 'llf_decrease_action', options=['ignore', 'warn', 'revert'])
    disp = int(disp)
    s = self._s
    llf = []
    params = [start_params]
    init = None
    inits = [self.ssm.initialization]
    i = 0
    delta = 0
    terminate = False
    while i < maxiter and (not terminate) and (i < 1 or delta > tolerance):
        out = self._em_iteration(params[-1], init=init, mstep_method=mstep_method)
        new_llf = out[0].llf_obs.sum()
        if not em_initialization:
            self.update(out[1])
            switch_init = []
            T = self['transition']
            init = self.ssm.initialization
            iloc = np.arange(self.k_states)
            if self.k_endog_Q == 0 and (not self.idiosyncratic_ar1):
                block = s.factor_blocks[0]
                if init.initialization_type == 'stationary':
                    Tb = T[block['factors'], block['factors']]
                    if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                        init.set(block['factors'], 'diffuse')
                        switch_init.append(f'factor block: {tuple(block.factor_names)}')
            else:
                for block in s.factor_blocks:
                    b = tuple(iloc[block['factors']])
                    init_type = init.blocks[b].initialization_type
                    if init_type == 'stationary':
                        Tb = T[block['factors'], block['factors']]
                        if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                            init.set(block['factors'], 'diffuse')
                            switch_init.append(f'factor block: {tuple(block.factor_names)}')
            if self.idiosyncratic_ar1:
                endog_names = self._get_endog_names(as_string=True)
                for j in range(s['idio_ar_M'].start, s['idio_ar_M'].stop):
                    init_type = init.blocks[j,].initialization_type
                    if init_type == 'stationary':
                        if not np.abs(T[j, j]) < 1 - 1e-10:
                            init.set(j, 'diffuse')
                            name = endog_names[j - s['idio_ar_M'].start]
                            switch_init.append(f'idiosyncratic AR(1) for monthly variable: {name}')
                if self.k_endog_Q > 0:
                    b = tuple(iloc[s['idio_ar_Q']])
                    init_type = init.blocks[b].initialization_type
                    if init_type == 'stationary':
                        Tb = T[s['idio_ar_Q'], s['idio_ar_Q']]
                        if not np.all(np.linalg.eigvals(Tb) < 1 - 1e-10):
                            init.set(s['idio_ar_Q'], 'diffuse')
                            switch_init.append('idiosyncratic AR(1) for the block of quarterly variables')
            if len(switch_init) > 0:
                warn(f'Non-stationary parameters found at EM iteration {i + 1}, which is not compatible with stationary initialization. Initialization was switched to diffuse for the following:  {switch_init}, and fitting was restarted.')
                results = self.fit_em(start_params=params[-1], transformed=transformed, cov_type=cov_type, cov_kwds=cov_kwds, maxiter=maxiter, tolerance=tolerance, em_initialization=em_initialization, mstep_method=mstep_method, full_output=full_output, disp=disp, return_params=return_params, low_memory=low_memory, llf_decrease_action=llf_decrease_action, llf_decrease_tolerance=llf_decrease_tolerance)
                self.ssm.initialize(self._default_initialization())
                return results
        llf_decrease = i > 0 and new_llf - llf[-1] < -llf_decrease_tolerance
        if llf_decrease_action == 'revert' and llf_decrease:
            warn(f'Log-likelihood decreased at EM iteration {i + 1}. Reverting to the results from EM iteration {i} (prior to the decrease) and returning the solution.')
            i -= 1
            terminate = True
        else:
            if llf_decrease_action == 'warn' and llf_decrease:
                warn(f'Log-likelihood decreased at EM iteration {i + 1}, which can indicate numerical issues.')
            llf.append(new_llf)
            params.append(out[1])
            if em_initialization:
                init = initialization.Initialization(self.k_states, 'known', constant=out[0].smoothed_state[..., 0], stationary_cov=out[0].smoothed_state_cov[..., 0])
                inits.append(init)
            if i > 0:
                delta = 2 * np.abs(llf[-1] - llf[-2]) / (np.abs(llf[-1]) + np.abs(llf[-2]))
            else:
                delta = np.inf
            if disp and i == 0:
                print(f'EM start iterations, llf={llf[-1]:.5g}')
            elif disp and (i + 1) % disp == 0:
                print(f'EM iteration {i + 1}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g}')
        i += 1
    not_converged = i == maxiter and delta > tolerance
    if not_converged:
        warn(f'EM reached maximum number of iterations ({maxiter}), without achieving convergence: llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
    if disp:
        if terminate:
            print(f'EM terminated at iteration {i}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
        elif not_converged:
            print(f'EM reached maximum number of iterations ({maxiter}), without achieving convergence: llf={llf[-1]:.5g}, convergence criterion={delta:.5g} (while specified tolerance was {tolerance:.5g})')
        else:
            print(f'EM converged at iteration {i}, llf={llf[-1]:.5g}, convergence criterion={delta:.5g} < tolerance={tolerance:.5g}')
    if return_params:
        result = params[-1]
    else:
        if em_initialization:
            base_init = self.ssm.initialization
            self.ssm.initialization = init
        result = self.smooth(params[-1], transformed=True, cov_type=cov_type, cov_kwds=cov_kwds)
        if em_initialization:
            self.ssm.initialization = base_init
        if full_output:
            llf.append(result.llf)
            em_retvals = Bunch(**{'params': np.array(params), 'llf': np.array(llf), 'iter': i, 'inits': inits})
            em_settings = Bunch(**{'method': 'em', 'tolerance': tolerance, 'maxiter': maxiter})
        else:
            em_retvals = None
            em_settings = None
        result._results.mle_retvals = em_retvals
        result._results.mle_settings = em_settings
    return result