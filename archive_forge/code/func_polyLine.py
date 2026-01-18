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
def polyLine(self, p):
    assert len(p) >= 1, 'Polyline must have 1 or more points'
    if self._strokeColor != None:
        self.setColor(self._strokeColor)
        self.moveTo(p[0][0], p[0][1])
        for t in p[1:]:
            self.lineTo(t[0], t[1])
        self.code_append('stroke')