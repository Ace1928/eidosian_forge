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
def interpolateLinear(t, a, b):
    """Linear interpolation between a and b, with t typically in [0, 1]."""
    return a + t * (b - a)