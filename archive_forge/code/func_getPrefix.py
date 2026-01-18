from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def getPrefix(self, uri):
    if uri not in self.prefixes:
        self.prefixes[uri] = 'xn%d' % self.prefixCounter
        self.prefixCounter = self.prefixCounter + 1
    return self.prefixes[uri]