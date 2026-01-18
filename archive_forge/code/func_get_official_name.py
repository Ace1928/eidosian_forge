import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def get_official_name(self):
    assert self.c_name_with_marker.endswith('&')
    return self.c_name_with_marker[:-1]