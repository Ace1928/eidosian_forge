import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def getSeriesOrder(self):
    bs = getattr(self, 'seriesOrder', None)
    n = len(self.data)
    if not bs:
        R = [(ss,) for ss in range(n)]
    else:
        bars = self.bars
        unseen = set(range(n))
        lines = set()
        R = []
        for s in bs:
            g = {ss for ss in s if 0 <= ss <= n}
            gl = {ss for ss in g if bars.checkAttr(ss, 'isLine', False)}
            if gl:
                g -= gl
                lines |= gl
                unseen -= gl
            if g:
                R.append(tuple(g))
                unseen -= g
        if unseen:
            R.extend(((ss,) for ss in sorted(unseen)))
        if lines:
            R.extend(((ss,) for ss in sorted(lines)))
    self._seriesOrder = R