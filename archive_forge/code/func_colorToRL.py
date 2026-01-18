import os
import types
from math import *
from reportlab.graphics import shapes
from reportlab.lib import colors
from rdkit.sping.PDF import pdfmetrics, pidPDF
from rdkit.sping.pid import *
def colorToRL(color):
    if color != transparent:
        return colors.Color(color.red, color.green, color.blue)
    else:
        return None