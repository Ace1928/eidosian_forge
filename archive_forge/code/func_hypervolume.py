import warnings
import numpy
def hypervolume(pointset, ref):
    """Compute the absolute hypervolume of a *pointset* according to the
    reference point *ref*.
    """
    warnings.warn('Falling back to the python version of hypervolume module. Expect this to be very slow.', RuntimeWarning)
    hv = _HyperVolume(ref)
    return hv.compute(pointset)