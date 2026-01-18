import contextlib
import threading
@contextlib.contextmanager
def preemption_save_context():
    _preemption_save_context.enter_preemption_save_context()
    try:
        yield
    finally:
        _preemption_save_context.exit_preemption_save_context()