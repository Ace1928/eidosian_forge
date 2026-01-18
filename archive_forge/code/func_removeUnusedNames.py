from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
@staticmethod
def removeUnusedNames(ttFont):
    """Remove any name records which are not in NameID range 0-255 and not utilized
        within the font itself."""
    visitor = NameRecordVisitor()
    visitor.visit(ttFont)
    toDelete = set()
    for record in ttFont['name'].names:
        if record.nameID < 256:
            continue
        if record.nameID not in visitor.seen:
            toDelete.add(record.nameID)
    for nameID in toDelete:
        ttFont['name'].removeNames(nameID)
    return toDelete