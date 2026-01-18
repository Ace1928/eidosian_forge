from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def gotEntityReference(self, entityRef):
    if entityRef in SuxElementStream.entities:
        data = SuxElementStream.entities[entityRef]
        if isinstance(data, bytes):
            data = data.decode('ascii')
        self.currElem.addContent(data)