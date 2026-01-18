from suds.sax import Namespace
from suds.sax.text import Text
from suds.sudsobject import Object
@classmethod
def genprefix(cls, node, ns):
    """
        Generate a prefix.

        @param node: XML node on which the prefix will be used.
        @type node: L{sax.element.Element}
        @param ns: Namespace needing a unique prefix.
        @type ns: (prefix, URI)
        @return: I{ns} with a new prefix.
        @rtype: (prefix, URI)

        """
    for i in range(1, 1024):
        prefix = 'ns%d' % (i,)
        uri = node.resolvePrefix(prefix, default=None)
        if uri in (None, ns[1]):
            return (prefix, ns[1])
    raise Exception('auto prefix, exhausted')