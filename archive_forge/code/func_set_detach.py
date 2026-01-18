import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def set_detach():
    """Sets up a following call to daemonize() to detach from the foreground
    session, running this process in the background."""
    global _detach
    _detach = True