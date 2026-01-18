import functools
@functools.cached_property
def hostmask(self):
    return self.network.hostmask