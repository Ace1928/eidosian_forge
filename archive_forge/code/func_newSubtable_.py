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
def newSubtable_(self, chaining=True):
    subtablename = f'Context{self.subtable_type}'
    if chaining:
        subtablename = 'Chain' + subtablename
    st = getattr(ot, subtablename)()
    setattr(st, f'{self.subtable_type}Count', 0)
    setattr(st, f'{self.subtable_type}LookupRecord', [])
    return st