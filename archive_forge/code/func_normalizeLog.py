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
def normalizeLog(value, rangeMin, rangeMax):
    """Logarithmically normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation."""
    logMin = math.log(rangeMin)
    logMax = math.log(rangeMax)
    return (math.log(value) - logMin) / (logMax - logMin)