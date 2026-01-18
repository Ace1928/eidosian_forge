from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def computeSize(self):
    """Called at start of draw.  Sets various column widths"""
    self._talksVisible = self.getRelevantTalks(self.data)
    self._trackCount = len(self.getAllTracks())
    self.computeStartAndEndTimes()
    self._colLeftEdges = [self.x]
    if self.timeColWidth is None:
        w = self.width / (1 + self._trackCount)
        self._colWidths = [w] * (1 + self._trackCount)
        for i in range(self._trackCount):
            self._colLeftEdges.append(self._colLeftEdges[-1] + w)
    else:
        self._colWidths = [self.timeColWidth]
        w = (self.width - self.timeColWidth) / self._trackCount
        for i in range(self._trackCount):
            self._colWidths.append(w)
            self._colLeftEdges.append(self._colLeftEdges[-1] + w)