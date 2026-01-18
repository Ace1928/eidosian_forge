from .compat import escape
from .jsonify import encode
def make_ns(self, ns):
    """
        Returns the `lazily` created template namespace.
        """
    if self.namespace:
        val = {}
        val.update(self.namespace)
        val.update(ns)
        return val
    else:
        return ns