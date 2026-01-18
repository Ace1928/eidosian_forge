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
def set_change_event_tracers(pre_tracer=None, post_tracer=None):
    """ Set the global trait change event tracers.

    The global tracers are called whenever a trait change event is dispatched.
    There are two tracers: `pre_tracer` is called before the notification is
    sent; `post_tracer` is called after the notification is sent, even if the
    notification failed with an exception (in which case the `post_tracer` is
    called with a reference to the exception, then the exception is sent to
    the `notification_exception_handler`).

    The tracers should be a callable taking 5 arguments:
    ::
      tracer(obj, trait_name, old, new, handler)

    `obj` is the source object, on which trait `trait_name` was changed from
    value `old` to value `new`. `handler` is the function or method that will
    be notified of the change.

    The post-notification tracer also has a keyword argument, `exception`,
    that is `None` if no exception has been raised, and the a reference to the
    raise exception otherwise.
    ::
      post_tracer(obj, trait_name, old, new, handler, exception=None)

    Note that for static trait change listeners, `handler` is not a method, but
    rather the function before class creation, since this is the way Traits
    works at the moment.
    """
    global _pre_change_event_tracer
    global _post_change_event_tracer
    _pre_change_event_tracer = pre_tracer
    _post_change_event_tracer = post_tracer