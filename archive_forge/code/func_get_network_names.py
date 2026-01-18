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
def get_network_names(self) -> t.Optional[list[str]]:
    """Return a list of the network names the container is attached to."""
    if self.networks is None:
        return None
    return sorted(self.networks)