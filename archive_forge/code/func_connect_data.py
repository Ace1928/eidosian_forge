import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def connect_data(self, detailed_signal, handler, *data, **kwargs):
    """Connect a callback to the given signal with optional user data.

        :param str detailed_signal:
            A detailed signal to connect to.
        :param callable handler:
            Callback handler to connect to the signal.
        :param *data:
            Variable data which is passed through to the signal handler.
        :param GObject.ConnectFlags connect_flags:
            Flags used for connection options.
        :returns:
            A signal id which can be used with disconnect.
        """
    flags = kwargs.get('connect_flags', 0)
    if flags & GObjectModule.ConnectFlags.AFTER:
        connect_func = _gi.GObject.connect_after
    else:
        connect_func = _gi.GObject.connect
    if flags & GObjectModule.ConnectFlags.SWAPPED:
        if len(data) != 1:
            raise ValueError('Using GObject.ConnectFlags.SWAPPED requires exactly one argument for user data, got: %s' % [data])

        def new_handler(obj, *args):
            args = list(args)
            swap = args.pop()
            args = args + [obj]
            return handler(swap, *args)
    else:
        new_handler = handler
    return connect_func(self, detailed_signal, new_handler, *data)