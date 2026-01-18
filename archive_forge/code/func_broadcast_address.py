import functools
@functools.cached_property
def broadcast_address(self):
    return self._address_class(int(self.network_address) | int(self.hostmask))