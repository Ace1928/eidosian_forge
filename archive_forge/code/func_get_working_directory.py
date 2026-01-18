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
def get_working_directory(self) -> str:
    """Return the working directory for the host."""
    if not self.pwd:
        ssh = self.get_origin_controller_connection()
        stdout = ssh.run(['pwd'], capture=True)[0]
        if self.args.explain:
            return '/pwd'
        pwd = stdout.strip().splitlines()[-1]
        if not pwd.startswith('/'):
            raise Exception(f'Unexpected current working directory "{pwd}" from "pwd" command output:\n{stdout.strip()}')
        self.pwd = pwd
    return self.pwd