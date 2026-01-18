import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def fontHeight(self, font=None):
    if not font:
        font = self.defaultFont
    return font.size