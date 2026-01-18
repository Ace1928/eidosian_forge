import asyncio
import inspect
import os
import signal
import time
from functools import partial
from threading import Thread
import pytest
import zmq
import zmq.asyncio
@pytest.fixture
def sigalrm_timeout():
    """Set timeout using SIGALRM

    Avoids infinite hang in context.term for an unclean context,
    raising an error instead.
    """
    if not hasattr(signal, 'SIGALRM') or not test_timeout_seconds:
        return

    def _alarm_timeout(*args):
        raise TimeoutError(f'Test did not complete in {test_timeout_seconds} seconds')
    signal.signal(signal.SIGALRM, _alarm_timeout)
    signal.alarm(test_timeout_seconds)