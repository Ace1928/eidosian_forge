import functools
def supernet_of(self, other):
    """Return True if this network is a supernet of other."""
    return self._is_subnet_of(other, self)