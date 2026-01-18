from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def isEqualToNode(self, other):
    """
        Compare this element to C{other}.  If the C{nodeName}, C{namespace},
        C{attributes}, and C{childNodes} are all the same, return C{True},
        otherwise return C{False}.
        """
    return self.nodeName.lower() == other.nodeName.lower() and self.namespace == other.namespace and (self.attributes == other.attributes) and Node.isEqualToNode(self, other)