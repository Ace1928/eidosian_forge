from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def findPrefixes(self, uri, match='eq'):
    """
        Find all prefixes that have been mapped to a namespace URI.

        The local mapping is searched, then walks up the tree until it reaches
        the top, collecting all matches.

        @param uri: A namespace URI.
        @type uri: basestring
        @param match: A matching function L{Element.matcher}.
        @type match: basestring
        @return: A list of mapped prefixes.
        @rtype: [basestring,...]

        """
    result = []
    for item in list(self.nsprefixes.items()):
        if self.matcher[match](item[1], uri):
            prefix = item[0]
            result.append(prefix)
    for item in list(self.specialprefixes.items()):
        if self.matcher[match](item[1], uri):
            prefix = item[0]
            result.append(prefix)
    if self.parent is not None:
        result += self.parent.findPrefixes(uri, match)
    return result