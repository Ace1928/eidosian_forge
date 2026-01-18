import collections
from . import _constants as C
@property
def idx_to_token(self):
    """
        list of strs:  A list of indexed tokens where the list indices and the token indices are aligned.
        """
    return self._idx_to_token