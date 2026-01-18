import numpy as np
from scipy.special import comb
def mc2mnc(mc):
    """convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    x = _convert_to_multidim(mc)

    def _local_counts(mc):
        mean = mc[0]
        mc = [1] + list(mc)
        mc[1] = 0
        mnc = [1, mean]
        for nn, m in enumerate(mc[2:]):
            n = nn + 2
            mnc.append(0)
            for k in range(n + 1):
                mnc[n] += comb(n, k, exact=True) * mc[k] * mean ** (n - k)
        return mnc[1:]
    res = np.apply_along_axis(_local_counts, 0, x)
    return _convert_from_multidim(res)