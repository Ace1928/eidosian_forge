from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def applyns(self, ns):
    """
        Apply the namespace to this node.

        If the prefix is I{None} then this element's explicit namespace
        I{expns} is set to the URI defined by I{ns}. Otherwise, the I{ns} is
        simply mapped.

        @param ns: A namespace.
        @type ns: (I{prefix}, I{URI})

        """
    if ns is None:
        return
    if not isinstance(ns, (list, tuple)):
        raise Exception('namespace must be a list or a tuple')
    if ns[0] is None:
        self.expns = ns[1]
    else:
        self.prefix = ns[0]
        self.nsprefixes[ns[0]] = ns[1]