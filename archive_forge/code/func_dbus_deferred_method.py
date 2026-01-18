from functools import wraps
import inspect
import dbus
from . import defer, Deferred, DeferredException
def dbus_deferred_method(*args, **kwargs):
    """Export the decorated method on the D-Bus and handle a maybe
    returned Deferred.

    This decorator can be applied to methods in the same way as the
    @dbus.service.method method, but it correctly handles the case where
    the method returns a Deferred.

    This decorator was kindly taken from James Henstridge blog post and
    adopted:
    http://blogs.gnome.org/jamesh/2009/07/06/watching-iview-with-rygel/
    """

    def decorator(function):
        function = dbus.service.method(*args, **kwargs)(function)

        @wraps(function)
        def wrapper(*args, **kwargs):

            def ignore_none_callback(*cb_args):
                if cb_args == (None,):
                    dbus_callback()
                else:
                    dbus_callback(*cb_args)
            dbus_callback = kwargs.pop('_dbus_callback')
            dbus_errback = kwargs.pop('_dbus_errback')
            deferred = defer(function, *args, **kwargs)
            deferred.add_callback(ignore_none_callback)
            deferred.add_errback(lambda error: dbus_errback(error.value))
        wrapper._dbus_async_callbacks = ('_dbus_callback', '_dbus_errback')
        return wrapper
    return decorator