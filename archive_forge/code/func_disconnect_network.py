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
def disconnect_network(self, network: str) -> None:
    """Disconnect the container from the specified network."""
    docker_network_disconnect(self.args, self.container_id, network)