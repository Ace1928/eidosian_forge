import numpy as np
from scipy import stats
from scipy.special import factorial
from statsmodels.base.model import GenericLikelihoodModel
def predict_distribution(self, exog):
    """return frozen scipy.stats distribution with mu at estimated prediction
        """
    if not hasattr(self, 'result'):
        raise ValueError
    else:
        result = self.result
        params = result.params
        mu = np.exp(np.dot(exog, params))
        return stats.poisson(mu, loc=0)