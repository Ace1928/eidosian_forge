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
def clear_change_event_tracers():
    """ Clear the global trait change event tracer. """
    global _pre_change_event_tracer
    global _post_change_event_tracer
    _pre_change_event_tracer = None
    _post_change_event_tracer = None