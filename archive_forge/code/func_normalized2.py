import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
def normalized2(self, demeaned=True):
    """return a normalized distribution where sigma=corr



        second implementation for testing affine transformation
        """
    if demeaned:
        shift = -self.mean
    else:
        shift = self.mean * (1.0 / self.std_sigma - 1.0)
    return self.affine_transformed(shift, np.diag(1.0 / self.std_sigma))