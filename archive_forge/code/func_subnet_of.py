import functools
def subnet_of(self, other):
    """Return True if this network is a subnet of other."""
    return self._is_subnet_of(self, other)