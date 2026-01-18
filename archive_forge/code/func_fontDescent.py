import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def fontDescent(self, font=None):
    if not font:
        font = self.defaultFont
    fontname = self._findPostScriptFontName(font)
    return -pdfmetrics.ascent_descent[fontname][1] * 0.001 * font.size