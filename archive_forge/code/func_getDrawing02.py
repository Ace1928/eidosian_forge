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
def getDrawing02():
    """Various Line shapes.

    The lines are blue and their strokeWidth is 5 mm.
    One line has a strokeDashArray set to [5, 10, 15].
    """
    D = Drawing(400, 200)
    D.add(Line(50, 50, 300, 100, strokeColor=colors.blue, strokeWidth=0.5 * cm))
    D.add(Line(50, 100, 300, 50, strokeColor=colors.blue, strokeWidth=0.5 * cm, strokeDashArray=[5, 10, 15]))
    return D