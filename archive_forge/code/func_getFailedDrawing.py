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
def getFailedDrawing(funcName):
    """Generate a drawing in case something goes really wrong.

    This will create a drawing to be displayed whenever some
    other drawing could not be executed, because the generating
    function does something terribly wrong! The box contains
    an attention triangle, plus some error message.
    """
    D = Drawing(400, 200)
    points = [200, 170, 140, 80, 260, 80]
    D.add(Polygon(points, strokeWidth=0.5 * cm, strokeColor=colors.red, fillColor=colors.yellow))
    s = String(200, 40, "Error in generating function '%s'!" % funcName, textAnchor='middle')
    D.add(s)
    return D