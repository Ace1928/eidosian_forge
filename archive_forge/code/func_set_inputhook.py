import sys
import select
def set_inputhook(self, callback):
    """Set inputhook to callback."""
    self._callback = callback