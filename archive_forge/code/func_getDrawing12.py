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
def getDrawing12():
    """Text strings in a non-standard font.
    All that is required is to place the .afm and .pfb files
    on the font path given in rl_config.py,
    for example in reportlab/fonts/.
    """
    faceName = 'DarkGardenMK'
    D = Drawing(400, 200)
    for size in range(12, 36, 4):
        D.add(String(10 + size * 2, 10 + size * 2, 'Hello World', fontName=faceName, fontSize=size))
    return D