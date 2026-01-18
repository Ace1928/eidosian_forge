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
def set_no_chdir():
    """Sets that we do not chdir to "/"."""
    global _chdir
    _chdir = False