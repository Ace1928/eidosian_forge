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
def setColor(self, color):
    if self._color != color:
        self._color = color
        if color:
            if hasattr(color, 'cyan'):
                self.code_append('%s setcmykcolor' % fp_str(color.cyan, color.magenta, color.yellow, color.black))
            else:
                self.code_append('%s setrgbcolor' % fp_str(color.red, color.green, color.blue))