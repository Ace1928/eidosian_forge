from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
@staticmethod
def interpolateFromValuesAndScalars(values, scalars):
    """Interpolate from values and scalars coefficients.

        If the values are master-values, then the scalars should be
        fetched from getMasterScalars().

        If the values are deltas, then the scalars should be fetched
        from getScalars(); in which case this is the same as
        interpolateFromDeltasAndScalars().
        """
    v = None
    assert len(values) == len(scalars)
    for value, scalar in zip(values, scalars):
        if not scalar:
            continue
        contribution = value * scalar
        if v is None:
            v = contribution
        else:
            v += contribution
    return v