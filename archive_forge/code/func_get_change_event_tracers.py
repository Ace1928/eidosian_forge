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
def get_change_event_tracers():
    """ Get the currently active global trait change event tracers. """
    return (_pre_change_event_tracer, _post_change_event_tracer)