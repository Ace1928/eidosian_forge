import functools
@property
def is_multicast(self):
    """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is a multicast address.
            See RFC 2373 2.7 for details.

        """
    return self in self._constants._multicast_network