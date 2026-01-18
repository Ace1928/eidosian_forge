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
def get_session_container_name(args: CommonConfig, name: str) -> str:
    """Return the given container name with the current test session name applied to it."""
    return f'{name}-{args.session_name}'