import functools
def address_exclude(self, other):
    """Remove an address from a larger block.

        For example:

            addr1 = ip_network('192.0.2.0/28')
            addr2 = ip_network('192.0.2.1/32')
            list(addr1.address_exclude(addr2)) =
                [IPv4Network('192.0.2.0/32'), IPv4Network('192.0.2.2/31'),
                 IPv4Network('192.0.2.4/30'), IPv4Network('192.0.2.8/29')]

        or IPv6:

            addr1 = ip_network('2001:db8::1/32')
            addr2 = ip_network('2001:db8::1/128')
            list(addr1.address_exclude(addr2)) =
                [ip_network('2001:db8::1/128'),
                 ip_network('2001:db8::2/127'),
                 ip_network('2001:db8::4/126'),
                 ip_network('2001:db8::8/125'),
                 ...
                 ip_network('2001:db8:8000::/33')]

        Args:
            other: An IPv4Network or IPv6Network object of the same type.

        Returns:
            An iterator of the IPv(4|6)Network objects which is self
            minus other.

        Raises:
            TypeError: If self and other are of differing address
              versions, or if other is not a network object.
            ValueError: If other is not completely contained by self.

        """
    if not self._version == other._version:
        raise TypeError('%s and %s are not of the same version' % (self, other))
    if not isinstance(other, _BaseNetwork):
        raise TypeError('%s is not a network object' % other)
    if not other.subnet_of(self):
        raise ValueError('%s not contained in %s' % (other, self))
    if other == self:
        return
    other = other.__class__('%s/%s' % (other.network_address, other.prefixlen))
    s1, s2 = self.subnets()
    while s1 != other and s2 != other:
        if other.subnet_of(s1):
            yield s2
            s1, s2 = s1.subnets()
        elif other.subnet_of(s2):
            yield s1
            s1, s2 = s2.subnets()
        else:
            raise AssertionError('Error performing exclusion: s1: %s s2: %s other: %s' % (s1, s2, other))
    if s1 == other:
        yield s2
    elif s2 == other:
        yield s1
    else:
        raise AssertionError('Error performing exclusion: s1: %s s2: %s other: %s' % (s1, s2, other))