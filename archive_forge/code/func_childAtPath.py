from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def childAtPath(self, path):
    """
        Get a child at I{path} where I{path} is a (/) separated list of element
        names that are expected to be children.

        @param path: A (/) separated list of element names.
        @type path: basestring
        @return: The leaf node at the end of I{path}.
        @rtype: L{Element}

        """
    result = None
    node = self
    for name in path.split('/'):
        if not name:
            continue
        ns = None
        prefix, name = splitPrefix(name)
        if prefix is not None:
            ns = node.resolvePrefix(prefix)
        result = node.getChild(name, ns)
        if result is None:
            return
        node = result
    return result