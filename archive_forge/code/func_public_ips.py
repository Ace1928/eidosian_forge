from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
@_requires_ips
def public_ips() -> list[str]:
    """return the IP addresses for this machine that are visible to other machines"""
    return PUBLIC_IPS