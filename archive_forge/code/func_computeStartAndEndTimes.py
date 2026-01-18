from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def computeStartAndEndTimes(self):
    """Work out first and last times to display"""
    if self.startTime:
        self._startTime = self.startTime
    else:
        for title, speaker, trackId, day, start, duration in self._talksVisible:
            if self._startTime is None:
                self._startTime = start
            elif start < self._startTime:
                self._startTime = start
    if self.endTime:
        self._endTime = self.endTime
    else:
        for title, speaker, trackId, day, start, duration in self._talksVisible:
            if self._endTime is None:
                self._endTime = start + duration
            elif start + duration > self._endTime:
                self._endTime = start + duration