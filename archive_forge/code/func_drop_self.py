from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
def drop_self(spec):
    args, varargs, varkw, defaults = spec
    if args[0:1] == ['self']:
        args = args[1:]
    return inspect.ArgSpec(args, varargs, varkw, defaults)