from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def isEqualToCharacterData(self, n):
    return self.value == n.value