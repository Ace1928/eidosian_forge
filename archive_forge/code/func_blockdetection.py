import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
@contextmanager
def blockdetection(timeout):
    """Context that raises an exception if process is blocking.

    Uses ``SIGALRM`` to detect blocking functions.
    """
    if not timeout:
        yield
    else:
        old_handler = signals['ALRM']
        old_handler = None if old_handler == _on_blocking else old_handler
        signals['ALRM'] = _on_blocking
        try:
            yield signals.arm_alarm(timeout)
        finally:
            if old_handler:
                signals['ALRM'] = old_handler
            signals.reset_alarm()