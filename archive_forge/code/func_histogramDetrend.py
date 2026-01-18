import numpy as np
from ...metaarray import MetaArray
def histogramDetrend(data, window=500, bins=50, threshold=3.0, offsetOnly=False):
    """Linear detrend. Works by finding the most common value at the beginning and end of a trace, excluding outliers.
    If offsetOnly is True, then only the offset from the beginning of the trace is subtracted.
    """
    d1 = data.view(np.ndarray)
    d2 = [d1[:window], d1[-window:]]
    v = [0, 0]
    for i in [0, 1]:
        d3 = d2[i]
        stdev = d3.std()
        mask = abs(d3 - np.median(d3)) < stdev * threshold
        d4 = d3[mask]
        y, x = np.histogram(d4, bins=bins)
        ind = np.argmax(y)
        v[i] = 0.5 * (x[ind] + x[ind + 1])
    if offsetOnly:
        d3 = data.view(np.ndarray) - v[0]
    else:
        base = np.linspace(v[0], v[1], len(data))
        d3 = data.view(np.ndarray) - base
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        return MetaArray(d3, info=data.infoCopy())
    return d3