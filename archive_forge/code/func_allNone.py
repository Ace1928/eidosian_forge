from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def allNone(lst):
    return all((l is None for l in lst))