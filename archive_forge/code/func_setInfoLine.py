import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def setInfoLine(self, s):
    self.pdf.setTitle(s)