from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawPolygon(self, polygon):
    assert len(polygon.points) >= 2, 'Polyline must have 2 or more points'
    head, tail = (polygon.points[0:2], polygon.points[2:])
    path = self._canvas.beginPath()
    path.moveTo(head[0], head[1])
    for i in range(0, len(tail), 2):
        path.lineTo(tail[i], tail[i + 1])
    path.close()
    self._canvas.drawPath(path, stroke=self._stroke, fill=self._fill)