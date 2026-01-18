from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def scaleTime(self, theTime):
    """Return y-value corresponding to times given"""
    axisHeight = self.height - self.trackRowHeight
    proportionUp = (theTime - self._startTime) / (self._endTime - self._startTime)
    y = self.y + axisHeight - axisHeight * proportionUp
    return y