import numpy as np
from scipy.special import comb
def mnc2cum(mnc):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    """
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        mnc = [1] + list(mnc)
        kappa = [1]
        for nn, m in enumerate(mnc[1:]):
            n = nn + 1
            kappa.append(m)
            for k in range(1, n):
                num_ways = comb(n - 1, k - 1, exact=True)
                kappa[n] -= num_ways * kappa[k] * mnc[n - k]
        return kappa[1:]
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res)