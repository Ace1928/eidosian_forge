import functools
@property
def packed(self):
    """The binary representation of this address."""
    return v6_int_to_packed(self._ip)