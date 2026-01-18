import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, closed=0):
    codeline = '%s m %s curveto'
    data = (fp_str(x1, y1), fp_str(x2, y2, x3, y3, x4, y4))
    if self._fillColor != None:
        self.setColor(self._fillColor)
        self.code_append(codeline % data + ' eofill')
    if self._strokeColor != None:
        self.setColor(self._strokeColor)
        self.code_append(codeline % data + (closed and ' closepath' or '') + ' stroke')