from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def subList(truth, lst):
    assert len(truth) == len(lst)
    return [l for l, t in zip(lst, truth) if t]