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
def findMultilingualName(self, names, windows=True, mac=True, minNameID=0, ttFont=None):
    """Return the name ID of an existing multilingual name that
        matches the 'names' dictionary, or None if not found.

        'names' is a dictionary with the name in multiple languages,
        such as {'en': 'Pale', 'de': 'Blaß', 'de-CH': 'Blass'}.
        The keys can be arbitrary IETF BCP 47 language codes;
        the values are Unicode strings.

        If 'windows' is True, the returned name ID is guaranteed
        exist for all requested languages for platformID=3 and
        platEncID=1.
        If 'mac' is True, the returned name ID is guaranteed to exist
        for all requested languages for platformID=1 and platEncID=0.

        The returned name ID will not be less than the 'minNameID'
        argument.
        """
    reqNameSet = set()
    for lang, name in sorted(names.items()):
        if windows:
            windowsName = _makeWindowsName(name, None, lang)
            if windowsName is not None:
                reqNameSet.add((windowsName.string, windowsName.platformID, windowsName.platEncID, windowsName.langID))
        if mac:
            macName = _makeMacName(name, None, lang, ttFont)
            if macName is not None:
                reqNameSet.add((macName.string, macName.platformID, macName.platEncID, macName.langID))
    matchingNames = dict()
    for name in self.names:
        try:
            key = (name.toUnicode(), name.platformID, name.platEncID, name.langID)
        except UnicodeDecodeError:
            continue
        if key in reqNameSet and name.nameID >= minNameID:
            nameSet = matchingNames.setdefault(name.nameID, set())
            nameSet.add(key)
    for nameID, nameSet in sorted(matchingNames.items()):
        if nameSet == reqNameSet:
            return nameID
    return None