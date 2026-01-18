import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def get_exception_errno(e):
    """A lot of methods on Python socket objects raise socket.error, but that
    exception is documented as having two completely different forms of
    arguments: either a string or a (errno, string) tuple.  We only want the
    errno."""
    if isinstance(e.args, tuple):
        return e.args[0]
    else:
        return errno.EPROTO