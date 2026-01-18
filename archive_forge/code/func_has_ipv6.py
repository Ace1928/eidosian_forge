import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
def has_ipv6() -> bool:
    """Check whether IPv6 is enabled on this host."""
    if system() not in ('Linux', 'Windows'):
        return False
    if socket.has_ipv6:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as server, socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
                server.bind(('::1', 0))
                port = server.getsockname()[1]
                server.listen()
                client.connect(('::1', port))
                conn, _ = server.accept()
                client.sendall('abc'.encode())
                msg = conn.recv(3).decode()
                assert msg == 'abc'
            return True
        except OSError:
            pass
    return False