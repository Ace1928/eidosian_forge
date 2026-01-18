import functools
from itertools import tee
@functools.wraps(fget)
def fget_memoized(self):
    if not hasattr(self, attr_name):
        setattr(self, attr_name, fget(self))
    return getattr(self, attr_name)