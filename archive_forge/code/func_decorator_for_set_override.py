import sys
import fixtures
from functools import wraps
def decorator_for_set_override(wrapped_function):

    @wraps(wrapped_function)
    def _wrapper(*args, **kwargs):
        group = 'oslo_messaging_notifications'
        if args[0] == 'notification_driver':
            args = ('driver', args[1], group)
        elif args[0] == 'notification_transport_url':
            args = ('transport_url', args[1], group)
        elif args[0] == 'notification_topics':
            args = ('topics', args[1], group)
        return wrapped_function(*args, **kwargs)
    _wrapper.wrapped = wrapped_function
    return _wrapper