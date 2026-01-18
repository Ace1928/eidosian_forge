import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def drawLines(self, lineList, color=None, width=None, dash=None, **kwargs):
    """Draws several distinct lines, all with same color
            and width, efficiently"""
    if color:
        self._updateLineColor(color)
    if width:
        self._updateLineWidth(width)
    self.pdf.lines(lineList)
    if color:
        self._updateLineColor(self.defaultLineColor)
    if width:
        self._updateLineWidth(self.defaultLineWidth)