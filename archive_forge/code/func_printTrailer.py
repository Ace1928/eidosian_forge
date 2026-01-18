import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
def printTrailer(self):
    print('trailer')
    print('<< /Size %d /Root %d 0 R /Info %d 0 R>>' % (len(self.objects) + 1, 1, self.infopos))
    print('startxref')
    print(self.startxref)