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
def planOpticalSizeAxis(glyphSetFunc, axisLimits, sizes=None, samples=None, glyphs=None, designLimits=None, pins=None, sanitize=False):
    """Plan a optical-size (`opsz`) axis.

    sizes: A list of optical size values to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=sizes, and the appropriate
    arguments. See documenation for planAxis for more information.
    """
    if sizes is None:
        sizes = SIZES
    return planAxis(measureWeight, normalizeLog, interpolateLog, glyphSetFunc, 'opsz', axisLimits, values=sizes, samples=samples, glyphs=glyphs, designLimits=designLimits, pins=pins)