from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
def getIndexFormatClass(self, indexFormat):
    return eblc_sub_table_classes[indexFormat]