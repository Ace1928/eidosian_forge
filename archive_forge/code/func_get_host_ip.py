import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_host_ip():
    """
    Wrapper to handle the case where the hostname is not yet available.
    """
    global _host_ip
    if _host_ip is None:
        while _host_ip is None:
            with contextlib.suppress(Exception):
                _host_ip = socket.gethostbyname(get_host_name())
            time.sleep(0.2)
    return _host_ip