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
def get_podman_default_hostname() -> t.Optional[str]:
    """
    Return the default hostname of the Podman service.

    --format was added in podman 3.3.0, this functionality depends on its availability
    """
    hostname: t.Optional[str] = None
    try:
        stdout = raw_command(['podman', 'system', 'connection', 'list', '--format=json'], env=docker_environment(), capture=True)[0]
    except SubprocessError:
        stdout = '[]'
    try:
        connections = json.loads(stdout)
    except json.decoder.JSONDecodeError:
        return hostname
    for connection in connections:
        if connection['Name'][-1] == '*':
            hostname = connection['URI']
            break
    return hostname