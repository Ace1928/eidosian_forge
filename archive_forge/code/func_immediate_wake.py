import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def immediate_wake(self):
    """Causes the following call to self.block() to wake up immediately,
        without blocking."""
    self.timeout = 0