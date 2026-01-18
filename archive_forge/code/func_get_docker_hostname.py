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
def get_docker_hostname() -> str:
    """Return the hostname of the Docker service."""
    docker_host = os.environ.get('DOCKER_HOST')
    if docker_host and docker_host.startswith('tcp://'):
        try:
            hostname = urllib.parse.urlparse(docker_host)[1].split(':')[0]
            display.info('Detected Docker host: %s' % hostname, verbosity=1)
        except ValueError:
            hostname = 'localhost'
            display.warning('Could not parse DOCKER_HOST environment variable "%s", falling back to localhost.' % docker_host)
    else:
        hostname = 'localhost'
        display.info('Assuming Docker is available on localhost.', verbosity=1)
    return hostname