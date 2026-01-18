from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def fit_joblib(self, data_generator, fit_kwds, parallel_backend, init_kwds_generator=None):
    """Performs the distributed estimation in parallel using joblib

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
    from statsmodels.tools.parallel import parallel_func
    par, f, n_jobs = parallel_func(_helper_fit_partition, self.partitions)
    if parallel_backend is None and init_kwds_generator is None:
        results_l = par((f(self, pnum, endog, exog, fit_kwds) for pnum, (endog, exog) in enumerate(data_generator)))
    elif parallel_backend is not None and init_kwds_generator is None:
        with parallel_backend:
            results_l = par((f(self, pnum, endog, exog, fit_kwds) for pnum, (endog, exog) in enumerate(data_generator)))
    elif parallel_backend is None and init_kwds_generator is not None:
        tup_gen = enumerate(zip(data_generator, init_kwds_generator))
        results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for pnum, ((endog, exog), init_kwds) in tup_gen))
    elif parallel_backend is not None and init_kwds_generator is not None:
        tup_gen = enumerate(zip(data_generator, init_kwds_generator))
        with parallel_backend:
            results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for pnum, ((endog, exog), init_kwds) in tup_gen))
    return results_l