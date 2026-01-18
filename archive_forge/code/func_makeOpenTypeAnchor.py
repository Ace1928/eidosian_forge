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
def makeOpenTypeAnchor(self, location, anchor):
    """ast.Anchor --> otTables.Anchor"""
    if anchor is None:
        return None
    variable = False
    deviceX, deviceY = (None, None)
    if anchor.xDeviceTable is not None:
        deviceX = otl.buildDevice(dict(anchor.xDeviceTable))
    if anchor.yDeviceTable is not None:
        deviceY = otl.buildDevice(dict(anchor.yDeviceTable))
    for dim in ('x', 'y'):
        varscalar = getattr(anchor, dim)
        if not isinstance(varscalar, VariableScalar):
            continue
        if getattr(anchor, dim + 'DeviceTable') is not None:
            raise FeatureLibError("Can't define a device coordinate and variable scalar", location)
        default, device = self.makeVariablePos(location, varscalar)
        setattr(anchor, dim, default)
        if device is not None:
            if dim == 'x':
                deviceX = device
            else:
                deviceY = device
            variable = True
    otlanchor = otl.buildAnchor(anchor.x, anchor.y, anchor.contourpoint, deviceX, deviceY)
    if variable:
        otlanchor.Format = 3
    return otlanchor