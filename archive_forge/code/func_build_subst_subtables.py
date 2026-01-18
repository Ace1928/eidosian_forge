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
def build_subst_subtables(self, mapping, klass):
    substitutions = [{}]
    for key in mapping:
        if key[0] == self.SUBTABLE_BREAK_:
            substitutions.append({})
        else:
            substitutions[-1][key] = mapping[key]
    subtables = [klass(s) for s in substitutions]
    return subtables