import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def as_function_pointer(self):
    return FunctionPtrType(self.args, self.result, self.ellipsis, self.abi)