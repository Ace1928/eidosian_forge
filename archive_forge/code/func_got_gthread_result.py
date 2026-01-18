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
def got_gthread_result(gthread):
    if future.done():
        return
    try:
        result = gthread.wait()
        future.set_result(result)
    except GreenletExit:
        future.cancel()
    except BaseException as e:
        future.set_exception(e)