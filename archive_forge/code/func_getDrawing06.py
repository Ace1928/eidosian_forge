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
def getDrawing06():
    """This demonstrates all the basic shapes at once.

    There are no groups or references.
    Each solid shape should have a green fill.
    """
    green = colors.green
    D = Drawing(400, 200)
    D.add(Line(10, 10, 390, 190))
    D.add(Circle(100, 100, 20, fillColor=green))
    D.add(Circle(200, 100, 40, fillColor=green))
    D.add(Circle(300, 100, 30, fillColor=green))
    D.add(Wedge(330, 100, 40, -10, 40, fillColor=green))
    D.add(PolyLine([120, 10, 130, 20, 140, 10, 150, 20, 160, 10, 170, 20, 180, 10, 190, 20, 200, 10], fillColor=green))
    D.add(Polygon([300, 20, 350, 20, 390, 80, 300, 75, 330, 40], fillColor=green))
    D.add(Ellipse(50, 150, 40, 20, fillColor=green))
    D.add(Rect(120, 150, 60, 30, strokeWidth=10, strokeColor=colors.yellow, fillColor=green))
    D.add(Rect(220, 150, 60, 30, 10, 10, fillColor=green))
    D.add(String(10, 50, 'Basic Shapes', fillColor=colors.black, fontName='Helvetica'))
    return D