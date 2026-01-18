from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os
def get_lookup_(self, location, builder_class):
    if self.cur_lookup_ and type(self.cur_lookup_) == builder_class and (self.cur_lookup_.lookupflag == self.lookupflag_) and (self.cur_lookup_.markFilterSet == self.lookupflag_markFilterSet_):
        return self.cur_lookup_
    if self.cur_lookup_name_ and self.cur_lookup_:
        raise FeatureLibError('Within a named lookup block, all rules must be of the same lookup type and flag', location)
    self.cur_lookup_ = builder_class(self.font, location)
    self.cur_lookup_.lookupflag = self.lookupflag_
    self.cur_lookup_.markFilterSet = self.lookupflag_markFilterSet_
    self.lookups_.append(self.cur_lookup_)
    if self.cur_lookup_name_:
        self.named_lookups_[self.cur_lookup_name_] = self.cur_lookup_
    if self.cur_feature_name_:
        self.add_lookup_to_feature_(self.cur_lookup_, self.cur_feature_name_)
    return self.cur_lookup_