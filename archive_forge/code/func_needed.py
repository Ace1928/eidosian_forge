import threading
from oslo_utils import timeutils
@property
def needed(self):
    """Returns how many decrements are needed before latch is released."""
    return max(0, self._count)