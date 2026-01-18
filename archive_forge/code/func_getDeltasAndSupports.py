from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def getDeltasAndSupports(self, items, *, round=noRound):
    model, items = self.getSubModel(items)
    return (model.getDeltas(items, round=round), model.supports)