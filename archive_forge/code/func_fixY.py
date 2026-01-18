import os
import types
from math import *
from reportlab.graphics import shapes
from reportlab.lib import colors
from rdkit.sping.PDF import pdfmetrics, pidPDF
from rdkit.sping.pid import *
def fixY(self, y):
    return self.size[1] - y