from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def allEqualTo(ref, lst, mapper=None):
    if mapper is None:
        return all((ref == item for item in lst))
    mapped = mapper(ref)
    return all((mapped == mapper(item) for item in lst))