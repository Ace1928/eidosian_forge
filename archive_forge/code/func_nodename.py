from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def nodename(name: str, hostname: str) -> str:
    """Create node name from name/hostname pair."""
    return NODENAME_SEP.join((name, hostname))