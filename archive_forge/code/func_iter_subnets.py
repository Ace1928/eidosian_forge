import struct
import sys
def iter_subnets(self, prefixlen_diff=1, new_prefix=None):
    """The subnets which join to make the current subnet.

        In the case that self contains only one IP
        (self._prefixlen == 32 for IPv4 or self._prefixlen == 128
        for IPv6), return a list with just ourself.

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
        raise ValueError('prefix length diff %d is invalid for netblock %s' % (new_prefixlen, str(self)))
    first = IPNetwork('%s/%s' % (str(self.network), str(self._prefixlen + prefixlen_diff)), version=self._version)
    yield first
    current = first
    while True:
        broadcast = current.broadcast
        if broadcast == self.broadcast:
            return
        new_addr = IPAddress(int(broadcast) + 1, version=self._version)
        current = IPNetwork('%s/%s' % (str(new_addr), str(new_prefixlen)), version=self._version)
        yield current