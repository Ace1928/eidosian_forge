from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def schedule_task(self, name: str, callback: Callable[[], None], at: Tat=None, period: str | dt.timedelta=None, cron: Optional[str]=None, threaded: bool=False) -> None:
    """
        Schedules a task at a specific time or on a schedule.

        By default the starting time is immediate but may be
        overridden with the `at` keyword argument. The scheduling may
        be declared using the `period` argument or a cron expression
        (which requires the `croniter` library). Note that the `at`
        time should be in local time but if a callable is provided it
        must return a UTC time.

        Note that the scheduled callback must not be defined within a
        script served using `panel serve` because the script and all
        its contents are cleaned up when the user session is
        destroyed. Therefore the callback must be imported from a
        separate module or should be scheduled from a setup script
        (provided to `panel serve` using the `--setup` argument). Note
        also that scheduling is idempotent, i.e.  if a callback has
        already been scheduled under the same name subsequent calls
        will have no effect. This is ensured that even if you schedule
        a task from within your application code, the task is only
        scheduled once.

        Arguments
        ---------
        name: str
          Name of the scheduled task
        callback: callable
          Callback to schedule
        at: datetime.datetime, Iterator or callable
          Declares a time to schedule the task at. May be given as a
          datetime or an Iterator of datetimes in the local timezone
          declaring when to execute the task. Alternatively may also
          declare a callable which is given the current UTC time and
          must return a datetime also in UTC.
        period: str or datetime.timedelta
          The period between executions, may be expressed as a
          timedelta or a string:

            - Week:   '1w'
            - Day:    '1d'
            - Hour:   '1h'
            - Minute: '1m'
            - Second: '1s'

        cron: str
          A cron expression (requires croniter to parse)
        threaded: bool
          Whether the callback should be run on a thread (requires
          config.nthreads to be set).
        """
    if name in self._scheduled:
        if callback is not self._scheduled[name][1]:
            self.param.warning(f'A separate task was already scheduled under the name {name!r}. The new task will be ignored. If you want to replace the existing task cancel it with `state.cancel_task({name!r})` before adding adding a new task under the same name.')
        return
    if getattr(callback, '__module__', '').startswith('bokeh_app_'):
        raise RuntimeError('Cannot schedule a task from within the context of an application. Either import the task callback from a separate module or schedule the task from a setup script that you provide to `panel serve` using the --setup commandline argument.')
    if cron is None:
        if isinstance(period, str):
            period = parse_timedelta(period)

        def dgen():
            if isinstance(at, Iterator):
                while True:
                    new = next(at)
                    yield new.timestamp()
            elif callable(at):
                while True:
                    new = at(dt.datetime.utcnow())
                    if new is None:
                        raise StopIteration
                    yield new.replace(tzinfo=dt.timezone.utc).astimezone().timestamp()
            elif period is None:
                yield at.timestamp()
                raise StopIteration
            new_time = at or dt.datetime.now()
            while True:
                yield new_time.timestamp()
                new_time += period
        diter = dgen()
    else:
        from croniter import croniter
        base = dt.datetime.now() if at is None else at
        diter = croniter(cron, base)
    now = dt.datetime.now().timestamp()
    try:
        call_time_seconds = next(diter) - now
    except StopIteration:
        return
    self._scheduled[name] = (diter, callback)
    self._ioloop.call_later(delay=call_time_seconds, callback=partial(self._scheduled_cb, name, threaded))