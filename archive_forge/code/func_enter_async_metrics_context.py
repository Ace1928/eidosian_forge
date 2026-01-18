import contextlib
import threading
def enter_async_metrics_context(self):
    self._in_async_metrics_context = True