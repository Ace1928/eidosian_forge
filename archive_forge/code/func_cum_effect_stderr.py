import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def cum_effect_stderr(self, orth=False):
    return np.array([tsa.unvec(np.sqrt(np.diag(c))) for c in self.cum_effect_cov(orth=orth)])