from __future__ import annotations
import abc
import shlex
import tempfile
import typing as t
from .io import (
from .config import (
from .util import (
from .util_common import (
from .docker_util import (
from .ssh import (
from .become import (
def error_callback(ex: SubprocessError) -> None:
    """Error handler."""
    self.capture_log_details(ssh_logfile.name, ex)