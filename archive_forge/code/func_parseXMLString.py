from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def parseXMLString(st):
    """
    Parse an XML readable object.
    """
    return parseString(st, caseInsensitive=0, preserveCase=1)