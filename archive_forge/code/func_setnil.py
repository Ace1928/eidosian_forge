from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
def setnil(self, node, content):
    """
        Set the value of the I{node} to nill.
        @param node: A I{nil} node.
        @type node: L{Element}
        @param content: The content for which processing has ended.
        @type content: L{Object}
        """
    self.marshaller.setnil(node, content)