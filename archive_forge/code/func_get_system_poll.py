import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def get_system_poll():
    """Returns the original select.poll() object. If select.poll is
    monkey patched by eventlet or gevent library, it gets the original
    select.poll and returns an object of it using the
    eventlet.patcher.original/gevent.monkey.get_original functions.

    As a last resort, if there is any exception it returns the
    SelectPoll() object.
    """
    try:
        if _using_eventlet_green_select():
            _system_poll = eventlet_patcher.original('select').poll
        elif gevent_monkey and gevent_monkey.is_object_patched('select', 'poll'):
            _system_poll = gevent_monkey.get_original('select', 'poll')
        else:
            _system_poll = select.poll
    except:
        _system_poll = SelectPoll
    return _system_poll()