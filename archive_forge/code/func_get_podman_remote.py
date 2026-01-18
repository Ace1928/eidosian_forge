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
def get_podman_remote() -> t.Optional[str]:
    """Return the remote podman hostname, if any, otherwise return None."""
    hostname = None
    podman_host = os.environ.get('CONTAINER_HOST')
    if not podman_host:
        podman_host = get_podman_default_hostname()
    if podman_host and podman_host.startswith('ssh://'):
        try:
            hostname = urllib.parse.urlparse(podman_host).hostname
        except ValueError:
            display.warning('Could not parse podman URI "%s"' % podman_host)
        else:
            display.info('Detected Podman remote: %s' % hostname, verbosity=1)
    return hostname