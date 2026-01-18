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
def getAllTestDrawings(doTTF=1):
    D = []
    for f in getAllFunctionDrawingNames(doTTF=doTTF):
        _evalFuncDrawing(f, D)
    return D