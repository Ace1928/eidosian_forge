from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawEllipse(self, ellipse):
    x1 = ellipse.cx - ellipse.rx
    x2 = ellipse.cx + ellipse.rx
    y1 = ellipse.cy - ellipse.ry
    y2 = ellipse.cy + ellipse.ry
    self._canvas.ellipse(x1, y1, x2, y2, fill=self._fill, stroke=self._stroke)