import threading
import functools
import numba.core.event as ev
def require_global_compiler_lock():
    """Sentry that checks the global_compiler_lock is acquired.
    """
    assert global_compiler_lock.is_locked()