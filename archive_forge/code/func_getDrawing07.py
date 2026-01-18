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
def getDrawing07():
    """This tests the ability to translate and rotate groups.  The first set of axes should be
    near the bottom left of the drawing.  The second should be rotated counterclockwise
    by 15 degrees.  The third should be rotated by 30 degrees."""
    D = Drawing(400, 200)
    Axis = Group(Line(0, 0, 100, 0), Line(0, 0, 0, 50), Line(0, 10, 10, 10), Line(0, 20, 10, 20), Line(0, 30, 10, 30), Line(0, 40, 10, 40), Line(10, 0, 10, 10), Line(20, 0, 20, 10), Line(30, 0, 30, 10), Line(40, 0, 40, 10), Line(50, 0, 50, 10), Line(60, 0, 60, 10), Line(70, 0, 70, 10), Line(80, 0, 80, 10), Line(90, 0, 90, 10), String(20, 35, 'Axes', fill=colors.black))
    firstAxisGroup = Group(Axis)
    firstAxisGroup.translate(10, 10)
    D.add(firstAxisGroup)
    secondAxisGroup = Group(Axis)
    secondAxisGroup.translate(150, 10)
    secondAxisGroup.rotate(15)
    D.add(secondAxisGroup)
    thirdAxisGroup = Group(Axis, transform=mmult(translate(300, 10), rotate(30)))
    D.add(thirdAxisGroup)
    return D