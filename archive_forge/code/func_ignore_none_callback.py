from functools import wraps
import inspect
import dbus
from . import defer, Deferred, DeferredException
def ignore_none_callback(*cb_args):
    if cb_args == (None,):
        dbus_callback()
    else:
        dbus_callback(*cb_args)