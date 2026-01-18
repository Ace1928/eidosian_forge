from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def interpolateFromDeltas(self, loc, deltas):
    """Interpolate from deltas, at location loc."""
    scalars = self.getScalars(loc)
    return self.interpolateFromDeltasAndScalars(deltas, scalars)