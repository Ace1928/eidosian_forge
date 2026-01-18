from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def cleanupBox(box):
    """Return a sparse copy of `box`, without redundant (default) values.

    >>> cleanupBox({})
    {}
    >>> cleanupBox({'wdth': (0.0, 1.0)})
    {'wdth': (0.0, 1.0)}
    >>> cleanupBox({'wdth': (-1.0, 1.0)})
    {}

    """
    return {tag: limit for tag, limit in box.items() if limit != (-1.0, 1.0)}