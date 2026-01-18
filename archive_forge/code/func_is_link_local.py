import functools
@property
def is_link_local(self):
    """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is reserved per RFC 4291.

        """
    return self in self._constants._linklocal_network