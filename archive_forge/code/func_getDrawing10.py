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
def getDrawing10():
    """This tests nested groups with multiple levels of coordinate transformation.
    Each box should be staggered up and to the right, moving by 25 points each time."""
    D = Drawing(400, 200)
    fontName = _FONTS[0]
    fontSize = 12
    g1 = Group(Rect(0, 0, 100, 20, fillColor=colors.yellow), String(5, 5, 'Text in the box', fontName=fontName, fontSize=fontSize))
    D.add(g1)
    g2 = Group(g1, transform=translate(25, 25))
    D.add(g2)
    g3 = Group(g2, transform=translate(25, 25))
    D.add(g3)
    g4 = Group(g3, transform=translate(25, 25))
    D.add(g4)
    return D