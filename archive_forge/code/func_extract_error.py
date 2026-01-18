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
def extract_error(self, value: str) -> t.Optional[str]:
    """
        Extract the ansible-test portion of the error message from the given value and return it.
        Returns None if no ansible-test marker was found.
        """
    lines = value.strip().splitlines()
    try:
        idx = lines.index(self.MARKER)
    except ValueError:
        return None
    lines = lines[idx + 1:]
    message = '\n'.join(lines)
    return message