from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def promotePrefixes(self):
    """
        Push prefix declarations up the tree as far as possible.

        Prefix mapping are pushed to its parent unless the parent has the
        prefix mapped to another URI or the parent has the prefix. This is
        propagated up the tree until the top is reached.

        @return: self
        @rtype: L{Element}

        """
    for c in self.children:
        c.promotePrefixes()
    if self.parent is None:
        return
    for p, u in list(self.nsprefixes.items()):
        if p in self.parent.nsprefixes:
            pu = self.parent.nsprefixes[p]
            if pu == u:
                del self.nsprefixes[p]
            continue
        if p != self.parent.prefix:
            self.parent.nsprefixes[p] = u
            del self.nsprefixes[p]
    return self