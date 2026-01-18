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
def buildAlternateSubstSubtable(mapping):
    """Builds an alternate substitution (GSUB3) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.AlternateSubstBuilder` instead.

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.AlternateSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.AlternateSubst()
    self.alternates = dict(mapping)
    return self