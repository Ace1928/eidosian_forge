from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def interpolateFromMastersAndScalars(self, masterValues, scalars, *, round=noRound):
    """Interpolate from master-values, and scalars fetched from
        getScalars(), which is useful when you want to interpolate
        multiple master-values with the same location."""
    deltas = self.getDeltas(masterValues, round=round)
    return self.interpolateFromDeltasAndScalars(deltas, scalars)