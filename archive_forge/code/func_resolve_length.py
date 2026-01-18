import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def resolve_length(self, newlength):
    return ArrayType(self.item, newlength)