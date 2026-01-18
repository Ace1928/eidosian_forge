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
def buildLookup(subtables, flags=0, markFilterSet=None):
    """Turns a collection of rules into a lookup.

    A Lookup (as defined in the `OpenType Spec <https://docs.microsoft.com/en-gb/typography/opentype/spec/chapter2#lookupTbl>`__)
    wraps the individual rules in a layout operation (substitution or
    positioning) in a data structure expressing their overall lookup type -
    for example, single substitution, mark-to-base attachment, and so on -
    as well as the lookup flags and any mark filtering sets. You may import
    the following constants to express lookup flags:

    - ``LOOKUP_FLAG_RIGHT_TO_LEFT``
    - ``LOOKUP_FLAG_IGNORE_BASE_GLYPHS``
    - ``LOOKUP_FLAG_IGNORE_LIGATURES``
    - ``LOOKUP_FLAG_IGNORE_MARKS``
    - ``LOOKUP_FLAG_USE_MARK_FILTERING_SET``

    Args:
        subtables: A list of layout subtable objects (e.g.
            ``MultipleSubst``, ``PairPos``, etc.) or ``None``.
        flags (int): This lookup's flags.
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.

    Returns:
        An ``otTables.Lookup`` object or ``None`` if there are no subtables
        supplied.
    """
    if subtables is None:
        return None
    subtables = [st for st in subtables if st is not None]
    if not subtables:
        return None
    assert all((t.LookupType == subtables[0].LookupType for t in subtables)), 'all subtables must have the same LookupType; got %s' % repr([t.LookupType for t in subtables])
    self = ot.Lookup()
    self.LookupType = subtables[0].LookupType
    self.LookupFlag = flags
    self.SubTable = subtables
    self.SubTableCount = len(self.SubTable)
    if markFilterSet is not None:
        self.LookupFlag |= LOOKUP_FLAG_USE_MARK_FILTERING_SET
        assert isinstance(markFilterSet, int), markFilterSet
        self.MarkFilteringSet = markFilterSet
    else:
        assert self.LookupFlag & LOOKUP_FLAG_USE_MARK_FILTERING_SET == 0, 'if markFilterSet is None, flags must not set LOOKUP_FLAG_USE_MARK_FILTERING_SET; flags=0x%04x' % flags
    return self