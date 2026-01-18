import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
def find_display():
    display = 10
    while True:
        try:
            f = open(lock_path(display), 'w+')
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                raise
        except OSError:
            display += 1
            continue
        return (display, f)