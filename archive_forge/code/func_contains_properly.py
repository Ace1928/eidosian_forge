from .base import GEOSBase
from .prototypes import prepared as capi
def contains_properly(self, other):
    return capi.prepared_contains_properly(self.ptr, other.ptr)