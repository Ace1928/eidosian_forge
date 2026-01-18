from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def childrenAtPath(self, path):
    """
        Get a list of children at I{path} where I{path} is a (/) separated list
        of element names expected to be children.

        @param path: A (/) separated list of element names.
        @type path: basestring
        @return: The collection leaf nodes at the end of I{path}.
        @rtype: [L{Element},...]

        """
    parts = [p for p in path.split('/') if p]
    if len(parts) == 1:
        return self.getChildren(path)
    return self.__childrenAtPath(parts)