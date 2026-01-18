from .base import GEOSBase
from .prototypes import prepared as capi
def intersects(self, other):
    return capi.prepared_intersects(self.ptr, other.ptr)