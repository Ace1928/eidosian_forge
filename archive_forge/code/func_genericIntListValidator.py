import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def genericIntListValidator(values, validValues):
    """
    Generic. (Added at version 2.)
    """
    if not isinstance(values, (list, tuple)):
        return False
    valuesSet = set(values)
    validValuesSet = set(validValues)
    if valuesSet - validValuesSet:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    return True