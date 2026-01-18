from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def makeSubstitutionsHashable(conditionalSubstitutions):
    """Turn all the substitution dictionaries in sorted tuples of tuples so
    they are hashable, to detect duplicates so we don't write out redundant
    data."""
    allSubstitutions = set()
    condSubst = []
    for conditionSet, substitutionMaps in conditionalSubstitutions:
        substitutions = []
        for substitutionMap in substitutionMaps:
            subst = tuple(sorted(substitutionMap.items()))
            substitutions.append(subst)
            allSubstitutions.add(subst)
        condSubst.append((conditionSet, substitutions))
    return (condSubst, sorted(allSubstitutions))