import contextlib
import threading
def exit_preemption_save_context(self):
    self._in_preemption_save_context = False