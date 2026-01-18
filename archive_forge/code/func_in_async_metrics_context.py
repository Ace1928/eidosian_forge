import contextlib
import threading
def in_async_metrics_context(self):
    return self._in_async_metrics_context