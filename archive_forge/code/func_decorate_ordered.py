import abc
import functools
import inspect
import logging
import threading
import traceback
from oslo_config import cfg
from oslo_service import service
from oslo_utils import eventletutils
from oslo_utils import timeutils
from stevedore import driver
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
@staticmethod
def decorate_ordered(fn, state, after, reset_after):

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._reset_lock:
            if reset_after is not None and self._states[reset_after].complete:
                self.reset_states()
        states = self._states
        log_after = kwargs.pop('log_after', DEFAULT_LOG_AFTER)
        timeout = kwargs.pop('timeout', None)
        timeout_timer = None
        if timeout is not None:
            timeout_timer = timeutils.StopWatch(duration=timeout)
            timeout_timer.start()
        if after is not None:
            states[after].wait_for_completion(state, log_after, timeout_timer)
        states[state].run_once(lambda: fn(self, *args, **kwargs), log_after, timeout_timer)
    return wrapper