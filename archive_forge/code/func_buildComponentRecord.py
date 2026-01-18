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
def buildComponentRecord(anchors):
    """Builds a component record.

    As part of building mark-to-ligature positioning rules, you will need to
    define ``ComponentRecord`` objects, which contain "an array of offsets...
    to the Anchor tables that define all the attachment points used to attach
    marks to the component." This function builds the component record.

    Args:
        anchors: A list of ``otTables.Anchor`` objects or ``None``.

    Returns:
        A ``otTables.ComponentRecord`` object or ``None`` if no anchors are
        supplied.
    """
    if not anchors:
        return None
    self = ot.ComponentRecord()
    self.LigatureAnchor = anchors
    return self