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
def planSlantAxis(glyphSetFunc, axisLimits, slants=None, samples=None, glyphs=None, designLimits=None, pins=None, sanitize=False):
    """Plan a slant (`slnt`) axis.

    slants: A list slant angles to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=slants, and the appropriate
    arguments. See documenation for planAxis for more information.
    """
    if slants is None:
        slants = SLANTS
    return planAxis(measureSlant, normalizeDegrees, interpolateLinear, glyphSetFunc, 'slnt', axisLimits, values=slants, samples=samples, glyphs=glyphs, designLimits=designLimits, pins=pins, sanitizeFunc=sanitizeSlant if sanitize else None)