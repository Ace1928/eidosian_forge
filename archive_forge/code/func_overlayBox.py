from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def overlayBox(top, bot):
    """Overlays ``top`` box on top of ``bot`` box.

    Returns two items:

    * Box for intersection of ``top`` and ``bot``, or None if they don't intersect.
    * Box for remainder of ``bot``.  Remainder box might not be exact (since the
      remainder might not be a simple box), but is inclusive of the exact
      remainder.
    """
    intersection = {}
    intersection.update(top)
    intersection.update(bot)
    for axisTag in set(top) & set(bot):
        min1, max1 = top[axisTag]
        min2, max2 = bot[axisTag]
        minimum = max(min1, min2)
        maximum = min(max1, max2)
        if not minimum < maximum:
            return (None, bot)
        intersection[axisTag] = (minimum, maximum)
    remainder = dict(bot)
    extruding = False
    fullyInside = True
    for axisTag in top:
        if axisTag in bot:
            continue
        extruding = True
        fullyInside = False
        break
    for axisTag in bot:
        if axisTag not in top:
            continue
        min1, max1 = intersection[axisTag]
        min2, max2 = bot[axisTag]
        if min1 <= min2 and max2 <= max1:
            continue
        if extruding:
            return (intersection, bot)
        extruding = True
        fullyInside = False
        if min1 <= min2:
            minimum = max(max1, min2)
            maximum = max2
        elif max2 <= max1:
            minimum = min2
            maximum = min(min1, max2)
        else:
            return (intersection, bot)
        remainder[axisTag] = (minimum, maximum)
    if fullyInside:
        return (intersection, None)
    return (intersection, remainder)