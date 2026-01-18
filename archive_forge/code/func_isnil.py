from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def isnil(self):
    """
        Get whether the element is I{nil} as defined by having an
        I{xsi:nil="true"} attribute.

        @return: True if I{nil}, else False
        @rtype: boolean

        """
    nilattr = self.getAttribute('nil', ns=Namespace.xsins)
    return nilattr is not None and nilattr.getValue().lower() == 'true'