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
def create_hosts_entries(context: dict[str, ContainerAccess]) -> list[str]:
    """Return hosts entries for the specified context."""
    entries = []
    unique_id = uuid.uuid4()
    for container in context.values():
        if container.forwards:
            host_ip = '127.0.0.1'
        else:
            host_ip = container.host_ip
        entries.append('%s %s # ansible-test %s' % (host_ip, ' '.join(container.names), unique_id))
    return entries