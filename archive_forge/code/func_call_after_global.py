from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def call_after_global(seconds, func, *args, **kwargs):
    warnings.warn('call_after_global is renamed to spawn_after, whichhas the same signature and semantics (plus a bit extra).  Please do a quick search-and-replace on your codebase, thanks!', DeprecationWarning, stacklevel=2)
    return _spawn_n(seconds, func, args, kwargs)[0]