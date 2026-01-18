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
def docker_start(args: CommonConfig, container_id: str, options: list[str]) -> tuple[t.Optional[str], t.Optional[str]]:
    """Start a container by name or ID."""
    return docker_command(args, ['start'] + options + [container_id], capture=True)