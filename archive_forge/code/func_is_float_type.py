import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def is_float_type(self):
    return self.ALL_PRIMITIVE_TYPES[self.name] == 'f'