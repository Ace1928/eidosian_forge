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
def getDrawing11():
    """test of anchoring"""

    def makeSmiley(x, y, size, color):
        """Make a smiley data item representation."""
        d = size
        s = SmileyFace()
        s.fillColor = color
        s.x = x - d
        s.y = y - d
        s.size = d * 2
        return s
    D = Drawing(400, 200)
    g = Group(transform=(1, 0, 0, 1, 0, 0))
    g.add(makeSmiley(100, 100, 10, colors.red))
    g.add(Line(90, 100, 110, 100, strokeColor=colors.green))
    g.add(Line(100, 90, 100, 110, strokeColor=colors.green))
    D.add(g)
    g = Group(transform=(2, 0, 0, 2, 100, -100))
    g.add(makeSmiley(100, 100, 10, colors.blue))
    g.add(Line(90, 100, 110, 100, strokeColor=colors.green))
    g.add(Line(100, 90, 100, 110, strokeColor=colors.green))
    D.add(g)
    g = Group(transform=(2, 0, 0, 2, 0, 0))
    return D