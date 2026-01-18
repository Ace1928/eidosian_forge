from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def normalizeValue(v, triple, extrapolate=False):
    """Normalizes value based on a min/default/max triple.

    >>> normalizeValue(400, (100, 400, 900))
    0.0
    >>> normalizeValue(100, (100, 400, 900))
    -1.0
    >>> normalizeValue(650, (100, 400, 900))
    0.5
    """
    lower, default, upper = triple
    if not lower <= default <= upper:
        raise ValueError(f'Invalid axis values, must be minimum, default, maximum: {lower:3.3f}, {default:3.3f}, {upper:3.3f}')
    if not extrapolate:
        v = max(min(v, upper), lower)
    if v == default or lower == upper:
        return 0.0
    if v < default and lower != default or (v > default and upper == default):
        return (v - default) / (default - lower)
    else:
        assert v > default and upper != default or (v < default and lower == default), f'Ooops... v={v}, triple=({lower}, {default}, {upper})'
        return (v - default) / (upper - default)