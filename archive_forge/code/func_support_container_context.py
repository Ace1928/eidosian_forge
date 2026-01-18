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
@contextlib.contextmanager
def support_container_context(args: EnvironmentConfig, ssh: t.Optional[SshConnectionDetail]) -> c.Iterator[t.Optional[ContainerDatabase]]:
    """Create a context manager for integration tests that use support containers."""
    if not isinstance(args, (IntegrationConfig, UnitsConfig, SanityConfig, ShellConfig)):
        yield None
        return
    containers = get_container_database(args)
    if not containers.data:
        yield ContainerDatabase({})
        return
    context = create_support_container_context(args, ssh, containers)
    try:
        yield context.containers
    finally:
        context.close()