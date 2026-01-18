from __future__ import annotations
import dataclasses
import datetime
import functools
import os
import signal
import time
import typing as t
from .io import (
from .config import (
from .util import (
from .thread import (
from .constants import (
from .test import (
def configure_test_timeout(args: TestConfig) -> None:
    """Configure the test timeout."""
    timeout = get_timeout()
    if not timeout:
        return
    timeout_remaining = timeout.remaining
    test_timeout = TestTimeout(timeout.duration)
    if timeout_remaining <= datetime.timedelta():
        test_timeout.write(args)
        raise TimeoutExpiredError(f'The {timeout.duration} minute test timeout expired {timeout_remaining * -1} ago at {timeout.deadline}.')
    display.info(f'The {timeout.duration} minute test timeout expires in {timeout_remaining} at {timeout.deadline}.', verbosity=1)

    def timeout_handler(_dummy1: t.Any, _dummy2: t.Any) -> None:
        """Runs when SIGUSR1 is received."""
        test_timeout.write(args)
        raise TimeoutExpiredError(f'Tests aborted after exceeding the {timeout.duration} minute time limit.')

    def timeout_waiter(timeout_seconds: int) -> None:
        """Background thread which will kill the current process if the timeout elapses."""
        time.sleep(timeout_seconds)
        os.kill(os.getpid(), signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, timeout_handler)
    instance = WrappedThread(functools.partial(timeout_waiter, timeout_remaining.total_seconds()))
    instance.daemon = True
    instance.start()