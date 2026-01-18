import ctypes
import signal
import threading
def setup_death_penalty(self):
    """Starts the timer."""
    if self._timeout <= 0:
        return
    self._timer = self.new_timer()
    self._timer.start()