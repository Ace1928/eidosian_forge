from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def supportScalar(location, support, ot=True, extrapolate=False, axisRanges=None):
    """Returns the scalar multiplier at location, for a master
    with support.  If ot is True, then a peak value of zero
    for support of an axis means "axis does not participate".  That
    is how OpenType Variation Font technology works.

    If extrapolate is True, axisRanges must be a dict that maps axis
    names to (axisMin, axisMax) tuples.

      >>> supportScalar({}, {})
      1.0
      >>> supportScalar({'wght':.2}, {})
      1.0
      >>> supportScalar({'wght':.2}, {'wght':(0,2,3)})
      0.1
      >>> supportScalar({'wght':2.5}, {'wght':(0,2,4)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)}, ot=False)
      0.375
      >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':3}, {'wght':(0,1,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -1.0
      >>> supportScalar({'wght':-1}, {'wght':(0,1,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -1.0
      >>> supportScalar({'wght':3}, {'wght':(0,2,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      1.5
      >>> supportScalar({'wght':-1}, {'wght':(0,2,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -0.5
    """
    if extrapolate and axisRanges is None:
        raise TypeError('axisRanges must be passed when extrapolate is True')
    scalar = 1.0
    for axis, (lower, peak, upper) in support.items():
        if ot:
            if peak == 0.0:
                continue
            if lower > peak or peak > upper:
                continue
            if lower < 0.0 and upper > 0.0:
                continue
            v = location.get(axis, 0.0)
        else:
            assert axis in location
            v = location[axis]
        if v == peak:
            continue
        if extrapolate:
            axisMin, axisMax = axisRanges[axis]
            if v < axisMin and lower <= axisMin:
                if peak <= axisMin and peak < upper:
                    scalar *= (v - upper) / (peak - upper)
                    continue
                elif axisMin < peak:
                    scalar *= (v - lower) / (peak - lower)
                    continue
            elif axisMax < v and axisMax <= upper:
                if axisMax <= peak and lower < peak:
                    scalar *= (v - lower) / (peak - lower)
                    continue
                elif peak < axisMax:
                    scalar *= (v - upper) / (peak - upper)
                    continue
        if v <= lower or upper <= v:
            scalar = 0.0
            break
        if v < peak:
            scalar *= (v - lower) / (peak - lower)
        else:
            scalar *= (v - upper) / (peak - upper)
    return scalar