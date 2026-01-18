import numpy as np
from ...metaarray import MetaArray
def applyFilter(data, b, a, padding=100, bidir=True):
    """Apply a linear filter with coefficients a, b. Optionally pad the data before filtering
    and/or run the filter in both directions."""
    try:
        import scipy.signal
    except ImportError:
        raise Exception('applyFilter() requires the package scipy.signal.')
    d1 = data.view(np.ndarray)
    if padding > 0:
        d1 = np.hstack([d1[:padding], d1, d1[-padding:]])
    if bidir:
        d1 = scipy.signal.lfilter(b, a, scipy.signal.lfilter(b, a, d1)[::-1])[::-1]
    else:
        d1 = scipy.signal.lfilter(b, a, d1)
    if padding > 0:
        d1 = d1[padding:-padding]
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        return MetaArray(d1, info=data.infoCopy())
    else:
        return d1