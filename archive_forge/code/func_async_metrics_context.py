import contextlib
import threading
@contextlib.contextmanager
def async_metrics_context():
    _async_metrics_context.enter_async_metrics_context()
    try:
        yield
    finally:
        _async_metrics_context.exit_async_metrics_context()