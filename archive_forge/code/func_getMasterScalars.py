from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def getMasterScalars(self, targetLocation):
    """Return multipliers for each master, for the given location.
        If interpolating many master-values at the same location,
        this function allows speed up by fetching the scalars once
        and using them with interpolateFromValuesAndScalars().

        Note that the scalars used in interpolateFromMastersAndScalars(),
        are *not* the same as the ones returned here. They are the result
        of getScalars()."""
    out = self.getScalars(targetLocation)
    for i, weights in reversed(list(enumerate(self.deltaWeights))):
        for j, weight in weights.items():
            out[j] -= out[i] * weight
    out = [out[self.mapping[i]] for i in range(len(out))]
    return out