import functools
@property
def teredo(self):
    """Tuple of embedded teredo IPs.

        Returns:
            Tuple of the (server, client) IPs or None if the address
            doesn't appear to be a teredo address (doesn't start with
            2001::/32)

        """
    if self._ip >> 96 != 536936448:
        return None
    return (IPv4Address(self._ip >> 64 & 4294967295), IPv4Address(~self._ip & 4294967295))