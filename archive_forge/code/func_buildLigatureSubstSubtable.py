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
def buildLigatureSubstSubtable(mapping):
    """Builds a ligature substitution (GSUB4) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.LigatureSubstBuilder` instead.

    Example::

        # sub f f i by f_f_i;
        # sub f i by f_i;

        subtable = buildLigatureSubstSubtable({
            ("f", "f", "i"): "f_f_i",
            ("f", "i"): "f_i",
        })

    Args:
        mapping: A dictionary mapping tuples of glyph names to output
            glyph names.

    Returns:
        An ``otTables.LigatureSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.LigatureSubst()
    self.ligatures = {}
    for components in sorted(mapping.keys(), key=self._getLigatureSortKey):
        ligature = ot.Ligature()
        ligature.Component = components[1:]
        ligature.CompCount = len(ligature.Component) + 1
        ligature.LigGlyph = mapping[components]
        firstGlyph = components[0]
        self.ligatures.setdefault(firstGlyph, []).append(ligature)
    return self