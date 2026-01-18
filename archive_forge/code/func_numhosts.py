import struct
import sys
@property
def numhosts(self):
    """Number of hosts in the current subnet."""
    return int(self.broadcast) - int(self.network) + 1