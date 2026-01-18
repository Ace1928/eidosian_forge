import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
def loglike_leafbranch(self, params, tau):
    xb = self.xbetas(params)
    expxb = np.exp(xb / tau)
    sumexpxb = expxb.sum(1)
    logsumexpxb = np.log(sumexpxb)
    probs = expxb / sumexpxb[:, None]
    return (probs, logsumexpxp)