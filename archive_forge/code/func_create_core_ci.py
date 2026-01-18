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
def create_core_ci(self, load: bool) -> AnsibleCoreCI:
    """Create and return an AnsibleCoreCI instance."""
    if not self.config.arch:
        raise InternalError(f'No arch specified for config: {self.config}')
    return AnsibleCoreCI(args=self.args, resource=VmResource(platform=self.config.platform, version=self.config.version, architecture=self.config.arch, provider=self.config.provider, tag='controller' if self.controller else 'target'), load=load)