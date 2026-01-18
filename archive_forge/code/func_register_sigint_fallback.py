from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
@contextmanager
def register_sigint_fallback(callback):
    """Installs a SIGINT signal handler in case the default Python one is
    active which calls 'callback' in case the signal occurs.

    Only does something if called from the main thread.

    In case of nested context managers the signal handler will be only
    installed once and the callbacks will be called in the reverse order
    of their registration.

    The old signal handler will be restored in case no signal handler is
    registered while the context is active.
    """
    global _callback_stack, _sigint_called
    if not is_main_thread():
        yield
        return
    if not sigint_handler_is_default():
        if _callback_stack:
            _callback_stack.append(callback)
            try:
                yield
            finally:
                cb = _callback_stack.pop()
                if _sigint_called:
                    cb()
        else:
            yield
        return
    _sigint_called = False

    def sigint_handler(sig_num, frame):
        global _callback_stack, _sigint_called
        if _sigint_called:
            return
        _sigint_called = True
        _callback_stack.pop()()
    _callback_stack.append(callback)
    try:
        with sigint_handler_set_and_restore_default(sigint_handler):
            yield
    finally:
        if _sigint_called:
            signal.default_int_handler(signal.SIGINT, None)
        else:
            _callback_stack.pop()