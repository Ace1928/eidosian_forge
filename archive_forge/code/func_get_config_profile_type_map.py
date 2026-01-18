from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
@cache
def get_config_profile_type_map() -> dict[t.Type[HostConfig], t.Type[HostProfile]]:
    """Create and return a mapping of HostConfig types to HostProfile types."""
    return get_type_map(HostProfile, HostConfig)