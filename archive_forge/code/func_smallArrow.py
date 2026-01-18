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
def smallArrow():
    """create a small PIL image"""
    from reportlab.graphics.renderPM import _getImage
    b = base64_decodebytes(b'R0lGODdhCgAHAIMAAP/////29v/d3f+ysv9/f/9VVf9MTP8iIv8ICP8AAAAAAAAAAAAAAAAAAAAA\nAAAAACwAAAAACgAHAAAIMwABCBxIsKABAQASFli4MAECAgEAJJhIceKBAQkyasx4YECBjx8TICAQ\nAIDJkwYEAFgZEAA7')
    return _getImage().open(BytesIO(b))