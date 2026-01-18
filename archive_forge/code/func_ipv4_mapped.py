import functools
@property
def ipv4_mapped(self):
    """Return the IPv4 mapped address.

        Returns:
            If the IPv6 address is a v4 mapped address, return the
            IPv4 mapped address. Return None otherwise.

        """
    if self._ip >> 32 != 65535:
        return None
    return IPv4Address(self._ip & 4294967295)