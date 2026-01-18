from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
@staticmethod
def interpolateFromDeltasAndScalars(deltas, scalars):
    """Interpolate from deltas and scalars fetched from getScalars()."""
    return VariationModel.interpolateFromValuesAndScalars(deltas, scalars)