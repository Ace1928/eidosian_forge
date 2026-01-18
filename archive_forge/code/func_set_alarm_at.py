from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def set_alarm_at(self, tm: float, callback: Callable[[Self, _T], typing.Any], user_data: _T=None):
    """
        Schedule an alarm at *tm* time that will call *callback* from the
        within the :meth:`run` function. Returns a handle that may be passed to
        :meth:`remove_alarm`.

        :param tm: time to call callback e.g. ``time.time() + 5``
        :type tm: float
        :param callback: function to call with two parameters: this main loop
                         object and *user_data*
        :type callback: callable
        :param user_data: optional user data to pass to the callback
        :type user_data: object
        """
    sec = tm - time.time()
    self.logger.debug(f'Setting alarm in {sec!r} seconds with callback {callback!r}')

    def cb() -> None:
        callback(self, user_data)
    return self.event_loop.alarm(sec, cb)