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
def planWeightAxis(glyphSetFunc, axisLimits, weights=None, samples=None, glyphs=None, designLimits=None, pins=None, sanitize=False):
    """Plan a weight (`wght`) axis.

    weights: A list of weight values to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=weights, and the appropriate
    arguments. See documenation for planAxis for more information.
    """
    if weights is None:
        weights = WEIGHTS
    return planAxis(measureWeight, normalizeLinear, interpolateLog, glyphSetFunc, 'wght', axisLimits, values=weights, samples=samples, glyphs=glyphs, designLimits=designLimits, pins=pins, sanitizeFunc=sanitizeWeight if sanitize else None)