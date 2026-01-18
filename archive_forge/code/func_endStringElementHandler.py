import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def endStringElementHandler(self, tag):
    """Handle end of an XML string element."""
    element = self.element
    if element is not None:
        self.parser.StartElementHandler = self.startElementHandler
        self.parser.EndElementHandler = self.endElementHandler
        self.parser.CharacterDataHandler = self.skipCharacterDataHandler
    data = ''.join(self.data)
    self.data = []
    attributes = self.attributes
    self.attributes = None
    if self.namespace_prefix:
        try:
            uri, name = tag.split()
        except ValueError:
            pass
        else:
            prefix = self.namespace_prefix[uri]
            tag = f'{prefix}:{name}'
    if tag in self.items:
        assert tag == 'Item'
        key = attributes['Name']
        del attributes['Name']
    else:
        key = tag
    value = StringElement(data, tag, attributes, key)
    if element is None:
        self.record = element
    else:
        element.store(value)
    self.allowed_tags = None