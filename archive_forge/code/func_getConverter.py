from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def getConverter(self, featureTag):
    tableClass = self.featureParamTypes.get(featureTag, self.defaultFeatureParams)
    return self.__class__(self.name, self.repeat, self.aux, tableClass)