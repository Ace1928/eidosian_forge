from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def fit_sequential(self, data_generator, fit_kwds, init_kwds_generator=None):
    """Sequentially performs the distributed estimation using
        the corresponding DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
    results_l = []
    if init_kwds_generator is None:
        for pnum, (endog, exog) in enumerate(data_generator):
            results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds)
            results_l.append(results)
    else:
        tup_gen = enumerate(zip(data_generator, init_kwds_generator))
        for pnum, ((endog, exog), init_kwds_e) in tup_gen:
            results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e)
            results_l.append(results)
    return results_l