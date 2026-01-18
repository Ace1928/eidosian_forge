from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def getNamespaces(self):
    """
        Get the I{unique} set of namespaces referenced in the branch.

        @return: A set of namespaces.
        @rtype: set

        """
    s = set()
    for n in self.branch + self.node.ancestors():
        if self.permit(n.expns):
            s.add(n.expns)
        s = s.union(self.pset(n))
    return s