import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def as_raw_function(self):
    return RawFunctionType(self.args, self.result, self.ellipsis, self.abi)