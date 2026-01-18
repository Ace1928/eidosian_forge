import struct
import sys
def iterhosts(self):
    """Generate Iterator over usable hosts in a network.

           This is like __iter__ except it doesn't return the network
           or broadcast addresses.

        """
    cur = int(self.network) + 1
    bcast = int(self.broadcast) - 1
    while cur <= bcast:
        cur += 1
        yield IPAddress(cur - 1, version=self._version)