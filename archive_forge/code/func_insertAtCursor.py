import re
import string
from zope.interface import implementer
from incremental import Version
from twisted.conch.insults import insults
from twisted.internet import defer, protocol, reactor
from twisted.logger import Logger
from twisted.python import _textattributes
from twisted.python.compat import iterbytes
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
def insertAtCursor(b):
    f(b)
    del self.insertAtCursor
    self.activeCharset = oldActiveCharset