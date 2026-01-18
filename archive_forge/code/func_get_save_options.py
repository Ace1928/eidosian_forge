import contextlib
import threading
def get_save_options():
    """Returns the save options if under a save context."""
    return _save_context.options()