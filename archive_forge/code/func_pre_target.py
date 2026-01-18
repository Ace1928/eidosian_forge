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
def pre_target(target: IntegrationTarget) -> None:
    """Configure hosts for SSH port forwarding required by the specified target."""
    forward_ssh_ports(args, control_connections, '%s_hosts_prepare.yml' % control_type, control_state, target, HostType.control, control_contexts)
    forward_ssh_ports(args, managed_connections, '%s_hosts_prepare.yml' % managed_type, managed_state, target, HostType.managed, managed_contexts)