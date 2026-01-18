from suds import *
from suds.umx import *
from suds.umx.attrlist import AttrList
from suds.sax.text import Text
from suds.sudsobject import Factory, merge
def append_attributes(self, content):
    """
        Append attribute nodes into L{Content.data}.
        Attributes in the I{schema} or I{xml} namespaces are skipped.
        @param content: The current content being unmarshalled.
        @type content: L{Content}
        """
    attributes = AttrList(content.node.attributes)
    for attr in attributes.real():
        name = attr.name
        value = attr.value
        self.append_attribute(name, value, content)