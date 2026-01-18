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
def spawn_after_local(seconds, func, *args, **kwargs):
    """Spawns *func* after *seconds* have elapsed.  The function will NOT be
    called if the current greenthread has exited.

    *seconds* may be specified as an integer, or a float if fractional seconds
    are desired. The *func* will be called with the given *args* and
    keyword arguments *kwargs*, and will be executed within its own greenthread.

    The return value of :func:`spawn_after` is a :class:`GreenThread` object,
    which can be used to retrieve the results of the call.

    To cancel the spawn and prevent *func* from being called,
    call :meth:`GreenThread.cancel` on the return value. This will not abort the
    function if it's already started running.  If terminating *func* regardless
    of whether it's started or not is the desired behavior, call
    :meth:`GreenThread.kill`.
    """
    hub = hubs.get_hub()
    g = GreenThread(hub.greenlet)
    hub.schedule_call_local(seconds, g.switch, func, args, kwargs)
    return g