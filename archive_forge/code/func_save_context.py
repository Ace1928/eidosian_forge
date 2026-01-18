import contextlib
import threading
@contextlib.contextmanager
def save_context(options):
    if in_save_context():
        raise ValueError('Already in a SaveContext.')
    _save_context.enter_save_context(options)
    try:
        yield
    finally:
        _save_context.exit_save_context()