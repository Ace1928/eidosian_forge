import contextlib
import threading
def exit_save_context(self):
    self._in_save_context = False
    self._options = None