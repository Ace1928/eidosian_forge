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
def makeDesignspaceSnippet(axisTag, axisName, axisLimit, mapping):
    """Make a designspace snippet for a single axis."""
    designspaceSnippet = '    <axis tag="%s" name="%s" minimum="%g" default="%g" maximum="%g"' % ((axisTag, axisName) + axisLimit)
    if mapping:
        designspaceSnippet += '>\n'
    else:
        designspaceSnippet += '/>'
    for key, value in mapping.items():
        designspaceSnippet += '      <map input="%g" output="%g"/>\n' % (key, value)
    if mapping:
        designspaceSnippet += '    </axis>'
    return designspaceSnippet