import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def lr_effect_stderr(self, orth=False):
    cov = self.lr_effect_cov(orth=orth)
    return tsa.unvec(np.sqrt(np.diag(cov)))