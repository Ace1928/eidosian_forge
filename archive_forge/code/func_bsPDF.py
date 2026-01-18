from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def bsPDF(borderWidth, borderStyle, dashLen):
    d = dict(W=borderWidth, S=PDFName(_bsStyles[borderStyle]))
    if borderStyle == 'dashed':
        if not dashLen:
            dashLen = [3]
        elif not isinstance(dashLen, (list, tuple)):
            dashLen = [dashLen]
        d['D'] = PDFArray(dashLen)
    return PDFDictionary(d)