from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawRect(self, rect):
    if rect.rx == rect.ry == 0:
        self._canvas.rect(rect.x, rect.y, rect.width, rect.height, stroke=self._stroke, fill=self._fill)
    else:
        self._canvas.roundRect(rect.x, rect.y, rect.width, rect.height, rect.rx, fill=self._fill, stroke=self._stroke)