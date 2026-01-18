import functools
def subnets(self, prefixlen_diff=1, new_prefix=None):
    """The subnets which join to make the current subnet.

        In the case that self contains only one IP
        (self._prefixlen == 32 for IPv4 or self._prefixlen == 128
        for IPv6), yield an iterator with just ourself.

        Args:
            prefixlen_diff: An integer, the amount the prefix length
              should be increased by. This should not be set if
              new_prefix is also set.
            new_prefix: The desired new prefix length. This must be a
              larger number (smaller prefix) than the existing prefix.
              This should not be set if prefixlen_diff is also set.

        Returns:
            An iterator of IPv(4|6) objects.

        Raises:
            ValueError: The prefixlen_diff is too small or too large.
                OR
            prefixlen_diff and new_prefix are both set or new_prefix
              is a smaller number than the current prefix (smaller
              number means a larger network)

        """
    if self._prefixlen == self._max_prefixlen:
        yield self
        return
    if new_prefix is not None:
        if new_prefix < self._prefixlen:
            raise ValueError('new prefix must be longer')
        if prefixlen_diff != 1:
            raise ValueError('cannot set prefixlen_diff and new_prefix')
        prefixlen_diff = new_prefix - self._prefixlen
    if prefixlen_diff < 0:
        raise ValueError('prefix length diff must be > 0')
    new_prefixlen = self._prefixlen + prefixlen_diff
    if new_prefixlen > self._max_prefixlen:
        raise ValueError('prefix length diff %d is invalid for netblock %s' % (new_prefixlen, self))
    start = int(self.network_address)
    end = int(self.broadcast_address) + 1
    step = int(self.hostmask) + 1 >> prefixlen_diff
    for new_addr in range(start, end, step):
        current = self.__class__((new_addr, new_prefixlen))
        yield current