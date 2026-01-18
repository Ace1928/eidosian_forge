import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
def fitjoint(self):
    """fit a joint fixed effects model to all observations

        The regression results are attached as `lsjoint`.

        The contrasts for overall and pairwise tests for equality of coefficients are
        attached as a dictionary `contrasts`. This also includes the contrasts for the test
        that the coefficients of a level are zero. ::

        >>> res.contrasts.keys()
        [(0, 1), 1, 'all', 3, (1, 2), 2, (1, 3), (2, 3), (0, 3), (0, 2)]

        The keys are based on the original names or labels of the groups.

        TODO: keys can be numpy scalars and then the keys cannot be sorted



        """
    if not hasattr(self, 'weights'):
        self.fitbygroups()
    groupdummy = (self.groupsint[:, None] == self.uniqueint).astype(int)
    dummyexog = self.exog[:, None, :] * groupdummy[:, 1:, None]
    exog = np.c_[self.exog, dummyexog.reshape(self.exog.shape[0], -1)]
    if self.het:
        weights = self.weights
        res = WLS(self.endog, exog, weights=weights).fit()
    else:
        res = OLS(self.endog, exog).fit()
    self.lsjoint = res
    contrasts = {}
    nvars = self.exog.shape[1]
    nparams = exog.shape[1]
    ndummies = nparams - nvars
    contrasts['all'] = np.c_[np.zeros((ndummies, nvars)), np.eye(ndummies)]
    for groupind, group in enumerate(self.unique[1:]):
        groupind = groupind + 1
        contr = np.zeros((nvars, nparams))
        contr[:, nvars * groupind:nvars * (groupind + 1)] = np.eye(nvars)
        contrasts[group] = contr
        contrasts[self.unique[0], group] = contr
    pairs = np.triu_indices(len(self.unique), 1)
    for ind1, ind2 in zip(*pairs):
        if ind1 == 0:
            continue
        g1 = self.unique[ind1]
        g2 = self.unique[ind2]
        group = (g1, g2)
        contr = np.zeros((nvars, nparams))
        contr[:, nvars * ind1:nvars * (ind1 + 1)] = np.eye(nvars)
        contr[:, nvars * ind2:nvars * (ind2 + 1)] = -np.eye(nvars)
        contrasts[group] = contr
    self.contrasts = contrasts