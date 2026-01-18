from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def findFeatureVariationRecord(featureVariations, conditionTable):
    """Find a FeatureVariationRecord that has the same conditionTable."""
    if featureVariations.Version != 65536:
        raise VarLibError(f'Unsupported FeatureVariations table version: 0x{featureVariations.Version:08x} (expected 0x00010000).')
    for fvr in featureVariations.FeatureVariationRecord:
        if conditionTable == fvr.ConditionSet.ConditionTable:
            return fvr
    return None