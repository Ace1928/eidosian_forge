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
def drawFigure(self, partList, closed=0):
    figureCode = []
    a = figureCode.append
    first = 1
    for part in partList:
        op = part[0]
        args = list(part[1:])
        if op == figureLine:
            if first:
                first = 0
                a('%s m' % fp_str(args[:2]))
            else:
                a('%s l' % fp_str(args[:2]))
            a('%s l' % fp_str(args[2:]))
        elif op == figureArc:
            first = 0
            x1, y1, x2, y2, startAngle, extent = args[:6]
            a(self._genArcCode(x1, y1, x2, y2, startAngle, extent))
        elif op == figureCurve:
            if first:
                first = 0
                a('%s m' % fp_str(args[:2]))
            else:
                a('%s l' % fp_str(args[:2]))
            a('%s curveto' % fp_str(args[2:]))
        else:
            raise TypeError('unknown figure operator: ' + op)
    if closed:
        a('closepath')
    self._fillAndStroke(figureCode)