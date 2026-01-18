import sys
from .compat import _winapi as win32  # noqa
def ensure_SemLock():
    try:
        from _billiard import SemLock
    except ImportError:
        try:
            from _multiprocessing import SemLock
        except ImportError:
            raise ImportError('This platform lacks a functioning sem_open implementation, therefore,\nthe required synchronization primitives needed will not function,\nsee issue 3770.')