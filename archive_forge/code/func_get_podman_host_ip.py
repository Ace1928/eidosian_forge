from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
@cache
def get_podman_host_ip() -> str:
    """Return the IP of the Podman host."""
    podman_host_ip = socket.gethostbyname(get_podman_hostname())
    display.info('Detected Podman host IP: %s' % podman_host_ip, verbosity=1)
    return podman_host_ip