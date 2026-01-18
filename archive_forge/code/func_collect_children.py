import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
def collect_children(self, *, blocking=False):
    """Internal routine to wait for children that have exited."""
    if self.active_children is None:
        return
    while len(self.active_children) >= self.max_children:
        try:
            pid, _ = os.waitpid(-1, 0)
            self.active_children.discard(pid)
        except ChildProcessError:
            self.active_children.clear()
        except OSError:
            break
    for pid in self.active_children.copy():
        try:
            flags = 0 if blocking else os.WNOHANG
            pid, _ = os.waitpid(pid, flags)
            self.active_children.discard(pid)
        except ChildProcessError:
            self.active_children.discard(pid)
        except OSError:
            pass