from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def getElementsByTagNameNoCase(iNode, name):
    name = name.lower()
    matches = []
    matches_append = matches.append
    slice = [iNode]
    while len(slice) > 0:
        c = slice.pop(0)
        if c.nodeName.lower() == name:
            matches_append(c)
        slice[:0] = c.childNodes
    return matches