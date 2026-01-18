import contextlib
import threading
def in_load_context(self):
    return self._entered_load_context