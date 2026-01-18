from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
def root_ssh(ssh: SshConnection) -> SshConnectionDetail:
    """Return the SSH connection details from the given SSH connection. If become was specified, the user will be changed to `root`."""
    settings = ssh.settings.__dict__.copy()
    if ssh.become:
        settings.update(user='root')
    return SshConnectionDetail(**settings)