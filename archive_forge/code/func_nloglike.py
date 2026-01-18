import numpy as np
from scipy import stats, optimize, special
def nloglike(params):
    """negative loglikelihood function of binned data

        corresponds to multinomial
        """
    prob = np.diff(distfn.cdf(binedges, *params))
    return -(lnnobsfact + np.sum(freq * np.log(prob) - special.gammaln(freq + 1)))