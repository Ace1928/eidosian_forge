from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def clearPrefix(self, prefix):
    """
        Clear the specified prefix from the prefix mappings.

        @param prefix: A prefix to clear.
        @type prefix: basestring
        @return: self
        @rtype: L{Element}

        """
    if prefix in self.nsprefixes:
        del self.nsprefixes[prefix]
    return self