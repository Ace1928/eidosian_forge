import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def resetToDefaults(self):
    self._currentColor = self.defaultLineColor
    self._currentWidth = self.defaultLineWidth
    self._currentFont = self.defaultFont