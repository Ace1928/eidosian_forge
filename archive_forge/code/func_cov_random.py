import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def cov_random(self):
    return self.model.cov_random()