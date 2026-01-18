from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def isEqualToEntityReference(self, n):
    if not isinstance(n, EntityReference):
        return 0
    return self.eref == n.eref and self.nodeValue == n.nodeValue