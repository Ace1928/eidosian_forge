import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
def connect_to_new_process(self, fds):
    """Request forkserver to create a child process.

        Returns a pair of fds (status_r, data_w).  The calling process can read
        the child process's pid and (eventually) its returncode from status_r.
        The calling process should write to data_w the pickled preparation and
        process data.
        """
    self.ensure_running()
    if len(fds) + 4 >= MAXFDS_TO_SEND:
        raise ValueError('too many fds')
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(self._forkserver_address)
        parent_r, child_w = os.pipe()
        child_r, parent_w = os.pipe()
        allfds = [child_r, child_w, self._forkserver_alive_fd, resource_tracker.getfd()]
        allfds += fds
        try:
            reduction.sendfds(client, allfds)
            return (parent_r, parent_w)
        except:
            os.close(parent_r)
            os.close(parent_w)
            raise
        finally:
            os.close(child_r)
            os.close(child_w)