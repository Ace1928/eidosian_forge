import ctypes
import signal
import threading
def cancel_death_penalty(self):
    """Cancels the timer."""
    if self._timeout <= 0:
        return
    self._timer.cancel()
    self._timer = None