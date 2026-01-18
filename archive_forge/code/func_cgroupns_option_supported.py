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
@property
def cgroupns_option_supported(self) -> bool:
    """Return True if the `--cgroupns` option is supported, otherwise return False."""
    if self.engine == 'docker':
        return self.client_major_minor_version >= (20, 10) and self.server_major_minor_version >= (20, 10)
    raise NotImplementedError(self.engine)