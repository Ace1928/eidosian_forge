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
def set_monitor():
    """Sets up a following call to daemonize() to fork a supervisory process to
    monitor the daemon and restart it if it dies due to an error signal."""
    global _monitor
    _monitor = True