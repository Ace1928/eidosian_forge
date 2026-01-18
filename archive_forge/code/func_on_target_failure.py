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
def on_target_failure(self) -> None:
    """Executed during failure handling if this profile is a target."""
    display.info(f'Checking container "{self.container_name}" logs...')
    try:
        docker_logs(self.args, self.container_name)
    except SubprocessError as ex:
        display.error(str(ex))
    if self.config.cgroup != CGroupVersion.NONE:
        display.info(f'Checking container "{self.container_name}" systemd logs...')
        try:
            docker_exec(self.args, self.container_name, ['journalctl'], capture=False)
        except SubprocessError as ex:
            display.error(str(ex))
    display.error(f'Connection to container "{self.container_name}" failed. See logs and original error above.')