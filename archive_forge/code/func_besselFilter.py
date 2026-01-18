import numpy as np
from ...metaarray import MetaArray
def besselFilter(data, cutoff, order=1, dt=None, btype='low', bidir=True):
    """return data passed through bessel filter"""
    try:
        import scipy.signal
    except ImportError:
        raise Exception('besselFilter() requires the package scipy.signal.')
    if dt is None:
        try:
            tvals = data.xvals('Time')
            dt = (tvals[-1] - tvals[0]) / (len(tvals) - 1)
        except:
            dt = 1.0
    b, a = scipy.signal.bessel(order, cutoff * dt, btype=btype)
    return applyFilter(data, b, a, bidir=bidir)