from __future__ import annotations
import collections.abc as c
import dataclasses
import functools
import itertools
import os
import pickle
import sys
import time
import traceback
import typing as t
from .config import (
from .util import (
from .util_common import (
from .thread import (
from .host_profiles import (
from .pypi_proxy import (
def dispatch_jobs(jobs: list[tuple[HostProfile, WrappedThread]]) -> None:
    """Run the given profile job threads and wait for them to complete."""
    for profile, thread in jobs:
        thread.daemon = True
        thread.start()
    while any((thread.is_alive() for profile, thread in jobs)):
        time.sleep(1)
    failed = False
    connection_failures = 0
    for profile, thread in jobs:
        try:
            thread.wait_for_result()
        except HostConnectionError as ex:
            display.error(f'Host {profile.config} connection failed:\n{ex}')
            failed = True
            connection_failures += 1
        except ApplicationError as ex:
            display.error(f'Host {profile.config} job failed:\n{ex}')
            failed = True
        except Exception as ex:
            name = f'{('' if ex.__class__.__module__ == 'builtins' else ex.__class__.__module__ + '.')}{ex.__class__.__qualname__}'
            display.error(f'Host {profile.config} job failed:\nTraceback (most recent call last):\n{''.join(traceback.format_tb(ex.__traceback__)).rstrip()}\n{name}: {ex}')
            failed = True
    if connection_failures:
        raise HostConnectionError(f'Host job(s) failed, including {connection_failures} connection failure(s). See previous error(s) for details.')
    if failed:
        raise ApplicationError('Host job(s) failed. See previous error(s) for details.')