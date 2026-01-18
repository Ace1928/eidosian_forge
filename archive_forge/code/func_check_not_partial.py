import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def check_not_partial(self):
    if self.partial and (not self.partial_resolved):
        raise VerificationMissing(self._get_c_name())