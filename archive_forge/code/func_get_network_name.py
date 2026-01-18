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
def get_network_name(self) -> str:
    """Return the network name the container is attached to. Raises an exception if no network, or more than one, is attached."""
    networks = self.get_network_names()
    if not networks:
        raise ApplicationError('No network found for Docker container: %s.' % self.id)
    if len(networks) > 1:
        raise ApplicationError('Found multiple networks for Docker container %s instead of only one: %s' % (self.id, ', '.join(networks)))
    return networks[0]