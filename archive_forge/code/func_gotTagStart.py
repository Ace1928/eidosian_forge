from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def gotTagStart(self, name, attributes):
    defaultUri = None
    localPrefixes = {}
    attribs = {}
    uri = None
    for k, v in list(attributes.items()):
        if k.startswith('xmlns'):
            x, p = _splitPrefix(k)
            if x is None:
                defaultUri = v
            else:
                localPrefixes[p] = v
            del attributes[k]
    self.prefixStack.append(localPrefixes)
    if defaultUri is None:
        if len(self.defaultNsStack) > 0:
            defaultUri = self.defaultNsStack[-1]
        else:
            defaultUri = ''
    prefix, name = _splitPrefix(name)
    if prefix is None:
        uri = defaultUri
    else:
        uri = self.findUri(prefix)
    for k, v in attributes.items():
        p, n = _splitPrefix(k)
        if p is None:
            attribs[n] = v
        else:
            attribs[self.findUri(p), n] = unescapeFromXml(v)
    e = Element((uri, name), defaultUri, attribs, localPrefixes)
    self.defaultNsStack.append(defaultUri)
    if self.documentStarted:
        if self.currElem is None:
            self.currElem = e
        else:
            self.currElem = self.currElem.addChild(e)
    else:
        self.rootElem = e
        self.documentStarted = True
        self.DocumentStartEvent(e)