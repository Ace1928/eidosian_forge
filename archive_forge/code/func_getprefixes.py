from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def getprefixes(self):
    """Add prefixes for each namespace referenced by parameter types."""
    namespaces = []
    for l in (self.params, self.types):
        for t, r in l:
            ns = r.namespace()
            if ns[1] is None:
                continue
            if ns[1] in namespaces:
                continue
            if Namespace.xs(ns) or Namespace.xsd(ns):
                continue
            namespaces.append(ns[1])
            if t == r:
                continue
            ns = t.namespace()
            if ns[1] is None:
                continue
            if ns[1] in namespaces:
                continue
            namespaces.append(ns[1])
    i = 0
    if self.wsdl.options.sortNamespaces:
        namespaces.sort()
    for u in namespaces:
        p = self.nextprefix()
        ns = (p, u)
        self.prefixes.append(ns)