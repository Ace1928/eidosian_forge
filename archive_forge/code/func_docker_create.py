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
def docker_create(args: CommonConfig, image: str, options: list[str], cmd: list[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    """Create a container using the given docker image."""
    return docker_command(args, ['create'] + options + [image] + cmd, capture=True)