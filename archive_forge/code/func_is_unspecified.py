import functools
@property
def is_unspecified(self):
    return self._ip == 0 and self.network.is_unspecified