import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]