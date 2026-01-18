import os, sys
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import asNative, base64_decodebytes
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import *
import unittest
from reportlab.rl_config import register_reset
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def getDrawing13():
    """Test Various TTF Fonts"""

    def drawit(F, w=400, h=200, fontSize=12, slack=2, gap=5):
        D = Drawing(w, h)
        th = 2 * gap + fontSize * 1.2
        gh = gap + 0.2 * fontSize
        y = h
        maxx = 0
        for fontName in F:
            y -= th
            text = fontName + asNative(b': I should be totally horizontal and enclosed in a box and end in alphabetagamma \xc2\xa2\xc2\xa9\xc2\xae\xc2\xa3\xca\xa5\xd0\x96\xd6\x83\xd7\x90\xd9\x82\xe0\xa6\x95\xce\xb1\xce\xb2\xce\xb3')
            textWidth = stringWidth(text, fontName, fontSize)
            maxx = max(maxx, textWidth + 20)
            D.add(Group(Rect(8, y - gh, textWidth + 4, th, strokeColor=colors.red, strokeWidth=0.5, fillColor=colors.lightgrey), String(10, y, text, fontName=fontName, fontSize=fontSize)))
            y -= 5
        return (maxx, h - y + gap, D)
    maxx, maxy, D = drawit(_FONTS)
    if maxx > 400 or maxy > 200:
        _, _, D = drawit(_FONTS, maxx, maxy)
    return D