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
def create_host_profile(args: EnvironmentConfig, config: HostConfig, controller: bool) -> HostProfile:
    """Create and return a host profile from the given host configuration."""
    profile_type = get_config_profile_type_map()[type(config)]
    profile = profile_type(args=args, config=config, targets=args.targets if controller else None)
    return profile