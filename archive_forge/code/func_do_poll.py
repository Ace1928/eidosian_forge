import errno
import select
import sys
from functools import partial
def do_poll(t):
    if t is not None:
        t *= 1000
    return poll_obj.poll(t)