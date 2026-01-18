from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def shouldPreserveSpace(self):
    for edx in range(len(self.elementstack)):
        el = self.elementstack[-edx]
        if el.tagName == 'pre' or el.getAttribute('xml:space', '') == 'preserve':
            return 1
    return 0