from fontTools.ttLib import newTable
from fontTools.ttLib.tables._f_v_a_r import Axis as fvarAxis
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.basePen import NullPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.varLib.models import piecewiseLinearMap, normalizeValue
from fontTools.misc.cliTools import makeOutputFileName
import math
import logging
from pprint import pformat
def normalizeDegrees(value, rangeMin, rangeMax):
    """Angularly normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation."""
    tanMin = math.tan(math.radians(rangeMin))
    tanMax = math.tan(math.radians(rangeMax))
    return (math.tan(math.radians(value)) - tanMin) / (tanMax - tanMin)