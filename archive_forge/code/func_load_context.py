import contextlib
import threading
@contextlib.contextmanager
def load_context(load_options):
    _load_context.set_load_options(load_options)
    try:
        yield
    finally:
        _load_context.clear_load_options()