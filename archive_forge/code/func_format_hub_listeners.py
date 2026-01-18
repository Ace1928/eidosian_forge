import os
import sys
import linecache
import re
import inspect
def format_hub_listeners():
    """ Returns a formatted string of the current listeners on the current
    hub.  This can be useful in determining what's going on in the event system,
    especially when used in conjunction with :func:`hub_listener_stacks`.
    """
    from eventlet import hubs
    hub = hubs.get_hub()
    result = ['READERS:']
    for l in hub.get_readers():
        result.append(repr(l))
    result.append('WRITERS:')
    for l in hub.get_writers():
        result.append(repr(l))
    return os.linesep.join(result)