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
def docker_cp_to(args: CommonConfig, container_id: str, src: str, dst: str) -> None:
    """Copy a file to the specified container."""
    docker_command(args, ['cp', src, '%s:%s' % (container_id, dst)], capture=True)