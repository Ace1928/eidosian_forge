from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def getVariationNameIDs(varfont):
    used = []
    if 'fvar' in varfont:
        fvar = varfont['fvar']
        for axis in fvar.axes:
            used.append(axis.axisNameID)
        for instance in fvar.instances:
            used.append(instance.subfamilyNameID)
            if instance.postscriptNameID != 65535:
                used.append(instance.postscriptNameID)
    if 'STAT' in varfont:
        stat = varfont['STAT'].table
        for axis in stat.DesignAxisRecord.Axis if stat.DesignAxisRecord else ():
            used.append(axis.AxisNameID)
        for value in stat.AxisValueArray.AxisValue if stat.AxisValueArray else ():
            used.append(value.ValueNameID)
        elidedFallbackNameID = getattr(stat, 'ElidedFallbackNameID', None)
        if elidedFallbackNameID is not None:
            used.append(elidedFallbackNameID)
    return {nameID for nameID in used if nameID > 255}