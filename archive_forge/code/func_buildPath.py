from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
@classmethod
def buildPath(self, parent, path):
    """
        Build the specified path as a/b/c.

        Any missing intermediate nodes are built automatically.

        @param parent: A parent element on which the path is built.
        @type parent: I{Element}
        @param path: A simple path separated by (/).
        @type path: basestring
        @return: The leaf node of I{path}.
        @rtype: L{Element}

        """
    for tag in path.split('/'):
        child = parent.getChild(tag)
        if child is None:
            child = Element(tag, parent)
        parent = child
    return child