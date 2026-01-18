import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def meigh(self):
    evecs = self.v.T
    evals = self.s ** 2
    return (evals, evecs)