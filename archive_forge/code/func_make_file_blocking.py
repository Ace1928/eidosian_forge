from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import socket as pysocket
from ansible.module_utils.six import PY2
def make_file_blocking(file):
    fcntl.fcntl(file.fileno(), fcntl.F_SETFL, fcntl.fcntl(file.fileno(), fcntl.F_GETFL) & ~os.O_NONBLOCK)