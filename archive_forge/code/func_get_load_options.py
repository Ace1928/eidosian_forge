import contextlib
import threading
def get_load_options():
    """Returns the load options under a load context."""
    return _load_context.load_options()