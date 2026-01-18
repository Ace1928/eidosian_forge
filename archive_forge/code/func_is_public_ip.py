from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
@_requires_ips
def is_public_ip(ip: str) -> bool:
    """is `ip` a publicly visible address?"""
    return ip in PUBLIC_IPS