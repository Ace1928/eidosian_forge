import suds
from suds import *
from suds.sax import *
from suds.sax.attribute import Attribute
from suds.sax.document import Document
from suds.sax.element import Element
from suds.sax.text import Text
import sys
from xml.sax import make_parser, InputSource, ContentHandler
from xml.sax.handler import feature_external_ges
def mapPrefix(self, node, attribute):
    if attribute.name == 'xmlns':
        if len(attribute.value):
            node.expns = str(attribute.value)
        return True
    if attribute.prefix == 'xmlns':
        prefix = attribute.name
        node.nsprefixes[prefix] = str(attribute.value)
        return True
    return False