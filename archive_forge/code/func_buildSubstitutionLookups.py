from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def buildSubstitutionLookups(gsub, allSubstitutions, processLast=False):
    """Build the lookups for the glyph substitutions, return a dict mapping
    the substitution to lookup indices."""
    firstIndex = len(gsub.LookupList.Lookup) if processLast else 0
    lookupMap = {}
    for i, substitutionMap in enumerate(allSubstitutions):
        lookupMap[substitutionMap] = firstIndex + i
    if not processLast:
        shift = len(allSubstitutions)
        visitor = ShifterVisitor(shift)
        visitor.visit(gsub.FeatureList.FeatureRecord)
        visitor.visit(gsub.LookupList.Lookup)
    for i, subst in enumerate(allSubstitutions):
        substMap = dict(subst)
        lookup = buildLookup([buildSingleSubstSubtable(substMap)])
        if processLast:
            gsub.LookupList.Lookup.append(lookup)
        else:
            gsub.LookupList.Lookup.insert(i, lookup)
        assert gsub.LookupList.Lookup[lookupMap[subst]] is lookup
    gsub.LookupList.LookupCount = len(gsub.LookupList.Lookup)
    return lookupMap