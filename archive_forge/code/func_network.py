import struct
import sys
@property
def network(self):
    x = self._cache.get('network')
    if x is None:
        x = IPAddress(self._ip & int(self.netmask), version=self._version)
        self._cache['network'] = x
    return x