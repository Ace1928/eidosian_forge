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
def buildFeatureParams(self, tag):
    params = None
    if tag == 'size':
        params = otTables.FeatureParamsSize()
        params.DesignSize, params.SubfamilyID, params.RangeStart, params.RangeEnd = self.size_parameters_
        if tag in self.featureNames_ids_:
            params.SubfamilyNameID = self.featureNames_ids_[tag]
        else:
            params.SubfamilyNameID = 0
    elif tag in self.featureNames_:
        if not self.featureNames_ids_:
            pass
        else:
            assert tag in self.featureNames_ids_
            params = otTables.FeatureParamsStylisticSet()
            params.Version = 0
            params.UINameID = self.featureNames_ids_[tag]
    elif tag in self.cv_parameters_:
        params = otTables.FeatureParamsCharacterVariants()
        params.Format = 0
        params.FeatUILabelNameID = self.cv_parameters_ids_.get((tag, 'FeatUILabelNameID'), 0)
        params.FeatUITooltipTextNameID = self.cv_parameters_ids_.get((tag, 'FeatUITooltipTextNameID'), 0)
        params.SampleTextNameID = self.cv_parameters_ids_.get((tag, 'SampleTextNameID'), 0)
        params.NumNamedParameters = self.cv_num_named_params_.get(tag, 0)
        params.FirstParamUILabelNameID = self.cv_parameters_ids_.get((tag, 'ParamUILabelNameID_0'), 0)
        params.CharCount = len(self.cv_characters_[tag])
        params.Character = self.cv_characters_[tag]
    return params