from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def gotTagEnd(self, name):
    if self.rootElem is None:
        raise ParserError('Element closed after end of document.')
    prefix, name = _splitPrefix(name)
    if prefix is None:
        uri = self.defaultNsStack[-1]
    else:
        uri = self.findUri(prefix)
    if self.currElem is None:
        if self.rootElem.name != name or self.rootElem.uri != uri:
            raise ParserError('Mismatched root elements')
        self.DocumentEndEvent()
        self.rootElem = None
    else:
        if self.currElem.name != name or self.currElem.uri != uri:
            raise ParserError('Malformed element close')
        self.prefixStack.pop()
        self.defaultNsStack.pop()
        if self.currElem.parent is None:
            self.currElem.parent = self.rootElem
            self.ElementEvent(self.currElem)
            self.currElem = None
        else:
            self.currElem = self.currElem.parent