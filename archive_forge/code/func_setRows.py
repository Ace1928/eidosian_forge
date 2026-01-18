from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def setRows(self, dataRows, bitDepth=1, metrics=None, reverseBytes=False):
    if metrics is None:
        metrics = self.metrics
    if reverseBytes:
        dataRows = map(_reverseBytes, dataRows)
    self.imageData = bytesjoin(dataRows)