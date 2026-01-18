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
def daemonize_complete():
    """If daemonization is configured, then this function notifies the parent
    process that the child process has completed startup successfully."""
    _fork_notify_startup(_daemonize_fd)
    if _detach:
        if _chdir:
            os.chdir('/')
        _close_standard_fds()