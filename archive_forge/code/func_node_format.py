from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def node_format(s: str, name: str, **extra: dict) -> str:
    """Format worker node name (name@host.com)."""
    shortname, host = nodesplit(name)
    return host_format(s, host, shortname or NODENAME_DEFAULT, p=name, **extra)