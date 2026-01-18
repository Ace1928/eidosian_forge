from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def getRelevantTalks(self, talkList):
    """Scans for tracks actually used"""
    used = []
    for talk in talkList:
        title, speaker, trackId, day, hours, duration = talk
        assert trackId != 0, 'trackId must be None or 1,2,3... zero not allowed!'
        if day == self.day:
            if (self.startTime is None or hours + duration >= self.startTime) and (self.endTime is None or hours <= self.endTime):
                used.append(talk)
    return used