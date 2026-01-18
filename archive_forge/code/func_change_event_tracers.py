import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
@contextlib.contextmanager
def change_event_tracers(pre_tracer, post_tracer):
    """ Context manager to temporarily change the global event tracers. """
    old_pre_tracer, old_post_tracer = get_change_event_tracers()
    set_change_event_tracers(pre_tracer, post_tracer)
    try:
        yield
    finally:
        set_change_event_tracers(old_pre_tracer, old_post_tracer)