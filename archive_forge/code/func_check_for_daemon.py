import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
def check_for_daemon(limit=10, delay=0.1, timeout=0.1, port=TCP_GIT_PORT):
    """Check for a running TCP daemon.

    Defaults to checking 10 times with a delay of 0.1 sec between tries.

    Args:
      limit: Number of attempts before deciding no daemon is running.
      delay: Delay between connection attempts.
      timeout: Socket timeout for connection attempts.
      port: Port on which we expect the daemon to appear.
    Returns: A boolean, true if a daemon is running on the specified port,
        false if not.
    """
    for _ in range(limit):
        time.sleep(delay)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(delay)
        try:
            s.connect(('localhost', port))
            return True
        except socket.timeout:
            pass
        except OSError as e:
            if getattr(e, 'errno', False) and e.errno != errno.ECONNREFUSED:
                raise
            elif e.args[0] != errno.ECONNREFUSED:
                raise
        finally:
            s.close()
    return False