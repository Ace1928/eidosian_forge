from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
def post_idle(self, ts, current):
    """Called after :meth:`idle` by :meth:`tick`.
        """
    self._frames += 1
    self._fps_counter += 1
    self._duration_count += 1
    self._sleep_time += current - ts
    t_tot = current - self._duration_ts0
    if t_tot >= 1.0:
        self._events_duration = (t_tot - self._sleep_time) / float(self._duration_count)
        self._duration_ts0 = current
        self._sleep_time = self._duration_count = 0
    if self._last_fps_tick is None:
        self._last_fps_tick = current
    elif current - self._last_fps_tick > 1:
        d = float(current - self._last_fps_tick)
        self._fps = self._fps_counter / d
        self._rfps = self._rfps_counter
        self._last_fps_tick = current
        self._fps_counter = 0
        self._rfps_counter = 0
    self._process_events()
    return self._dt