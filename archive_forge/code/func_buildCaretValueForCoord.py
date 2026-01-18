from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildCaretValueForCoord(coord):
    self = ot.CaretValue()
    if isinstance(coord, tuple):
        self.Format = 3
        self.Coordinate, self.DeviceTable = coord
    else:
        self.Format = 1
        self.Coordinate = coord
    return self