import importlib  # noqa: F401
import sys
import threading
def hooks_equal(existing_hook, hook):
    if hasattr(existing_hook, '__name__') and hasattr(hook, '__name__'):
        return existing_hook.__name__ == hook.__name__
    else:
        return False