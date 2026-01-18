from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def compare_score_test(self, submodel):
    """
        Perform a score test for the given submodel against this model.

        Parameters
        ----------
        submodel : GEEResults instance
            A fitted GEE model that is a submodel of this model.

        Returns
        -------
        A dictionary with keys "statistic", "p-value", and "df",
        containing the score test statistic, its chi^2 p-value,
        and the degrees of freedom used to compute the p-value.

        Notes
        -----
        The score test can be performed without calling 'fit' on the
        larger model.  The provided submodel must be obtained from a
        fitted GEE.

        This method performs the same score test as can be obtained by
        fitting the GEE with a linear constraint and calling `score_test`
        on the results.

        References
        ----------
        Xu Guo and Wei Pan (2002). "Small sample performance of the score
        test in GEE".
        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
        """
    self.scaletype = submodel.model.scaletype
    submod = submodel.model
    if self.exog.shape[0] != submod.exog.shape[0]:
        msg = 'Model and submodel have different numbers of cases.'
        raise ValueError(msg)
    if self.exog.shape[1] == submod.exog.shape[1]:
        msg = 'Model and submodel have the same number of variables'
        warnings.warn(msg)
    if not isinstance(self.family, type(submod.family)):
        msg = 'Model and submodel have different GLM families.'
        warnings.warn(msg)
    if not isinstance(self.cov_struct, type(submod.cov_struct)):
        warnings.warn('Model and submodel have different GEE covariance structures.')
    if not np.equal(self.weights, submod.weights).all():
        msg = 'Model and submodel should have the same weights.'
        warnings.warn(msg)
    qm, qc = _score_test_submodel(self, submodel.model)
    if qm is None:
        msg = 'The provided model is not a submodel.'
        raise ValueError(msg)
    params_ex = np.dot(qm, submodel.params)
    cov_struct_save = self.cov_struct
    import copy
    cached_means_save = copy.deepcopy(self.cached_means)
    self.cov_struct = submodel.cov_struct
    self.update_cached_means(params_ex)
    _, score = self._update_mean_params()
    if score is None:
        msg = 'Singular matrix encountered in GEE score test'
        warnings.warn(msg, ConvergenceWarning)
        return None
    if not hasattr(self, 'ddof_scale'):
        self.ddof_scale = self.exog.shape[1]
    if not hasattr(self, 'scaling_factor'):
        self.scaling_factor = 1
    _, ncov1, cmat = self._covmat()
    score2 = np.dot(qc.T, score)
    try:
        amat = np.linalg.inv(ncov1)
    except np.linalg.LinAlgError:
        amat = np.linalg.pinv(ncov1)
    bmat_11 = np.dot(qm.T, np.dot(cmat, qm))
    bmat_22 = np.dot(qc.T, np.dot(cmat, qc))
    bmat_12 = np.dot(qm.T, np.dot(cmat, qc))
    amat_11 = np.dot(qm.T, np.dot(amat, qm))
    amat_12 = np.dot(qm.T, np.dot(amat, qc))
    try:
        ab = np.linalg.solve(amat_11, bmat_12)
    except np.linalg.LinAlgError:
        ab = np.dot(np.linalg.pinv(amat_11), bmat_12)
    score_cov = bmat_22 - np.dot(amat_12.T, ab)
    try:
        aa = np.linalg.solve(amat_11, amat_12)
    except np.linalg.LinAlgError:
        aa = np.dot(np.linalg.pinv(amat_11), amat_12)
    score_cov -= np.dot(bmat_12.T, aa)
    try:
        ab = np.linalg.solve(amat_11, bmat_11)
    except np.linalg.LinAlgError:
        ab = np.dot(np.linalg.pinv(amat_11), bmat_11)
    try:
        aa = np.linalg.solve(amat_11, amat_12)
    except np.linalg.LinAlgError:
        aa = np.dot(np.linalg.pinv(amat_11), amat_12)
    score_cov += np.dot(amat_12.T, np.dot(ab, aa))
    self.cov_struct = cov_struct_save
    self.cached_means = cached_means_save
    from scipy.stats.distributions import chi2
    try:
        sc2 = np.linalg.solve(score_cov, score2)
    except np.linalg.LinAlgError:
        sc2 = np.dot(np.linalg.pinv(score_cov), score2)
    score_statistic = np.dot(score2, sc2)
    score_df = len(score2)
    score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
    return {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}