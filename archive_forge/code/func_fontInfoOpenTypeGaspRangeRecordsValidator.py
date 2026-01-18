import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoOpenTypeGaspRangeRecordsValidator(value):
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True
    validBehaviors = [0, 1, 2, 3]
    dictPrototype = dict(rangeMaxPPEM=(int, True), rangeGaspBehavior=(list, True))
    ppemOrder = []
    for rangeRecord in value:
        if not genericDictValidator(rangeRecord, dictPrototype):
            return False
        ppem = rangeRecord['rangeMaxPPEM']
        behavior = rangeRecord['rangeGaspBehavior']
        ppemValidity = genericNonNegativeIntValidator(ppem)
        if not ppemValidity:
            return False
        behaviorValidity = genericIntListValidator(behavior, validBehaviors)
        if not behaviorValidity:
            return False
        ppemOrder.append(ppem)
    if ppemOrder != sorted(ppemOrder):
        return False
    return True