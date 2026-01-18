from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
@staticmethod
def stdColors(t, b, f):
    if isinstance(f, CMYKColor) or isinstance(t, CMYKColor) or isinstance(b, CMYKColor):
        return (t or CMYKColor(0, 0, 0, 0.9), b or CMYKColor(0, 0, 0, 0.9), f or CMYKColor(0.12, 0.157, 0, 0))
    else:
        return (t or Color(0.1, 0.1, 0.1), b or Color(0.1, 0.1, 0.1), f or Color(0.8, 0.843, 1))