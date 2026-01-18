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
def create_managed_contexts(control_contexts: dict[str, dict[str, ContainerAccess]]) -> dict[str, dict[str, ContainerAccess]]:
    """Create managed contexts from the given control contexts."""
    managed_contexts: dict[str, dict[str, ContainerAccess]] = {}
    for context_name, control_context in control_contexts.items():
        managed_context = managed_contexts[context_name] = {}
        for container_name, control_container in control_context.items():
            managed_context[container_name] = ContainerAccess(control_container.host_ip, control_container.names, None, dict(control_container.port_map()))
    return managed_contexts