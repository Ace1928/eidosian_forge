from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def checkForceBorder(self, x, y, width, height, forceBorder, shape, borderStyle, borderWidth, borderColor, fillColor):
    if forceBorder:
        canv = self.canv
        canv.saveState()
        canv.resetTransforms()
        if borderWidth != None:
            hbw = 0.5 * borderWidth
            canv.setLineWidth(borderWidth)
            canv.setStrokeColor(borderColor)
            s = 1
        else:
            s = hbw = 0
        width -= 2 * hbw
        height -= 2 * hbw
        x += hbw
        y += hbw
        canv.setFillColor(fillColor)
        if shape == 'square':
            canv.rect(x, y, width, height, stroke=s, fill=1)
        else:
            r = min(width, height) * 0.5
            canv.circle(x + r, y + r, r, stroke=s, fill=1)
        canv.restoreState()