import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def deviance(self, ML=False):
    """deviance defined as 2 times the negative loglikelihood

        """
    return -2 * self.logL(ML=ML)