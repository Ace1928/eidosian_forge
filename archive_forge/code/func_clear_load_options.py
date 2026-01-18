import contextlib
import threading
def clear_load_options(self):
    self._load_options = None
    self._entered_load_context.pop()