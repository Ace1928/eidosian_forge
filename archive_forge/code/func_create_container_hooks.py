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
def create_container_hooks(args: IntegrationConfig, control_connections: list[SshConnectionDetail], managed_connections: t.Optional[list[SshConnectionDetail]]) -> tuple[t.Optional[c.Callable[[IntegrationTarget], None]], t.Optional[c.Callable[[IntegrationTarget], None]]]:
    """Return pre and post target callbacks for enabling and disabling container access for each test target."""
    containers = get_container_database(args)
    control_contexts = containers.data.get(HostType.control)
    if control_contexts:
        managed_contexts = containers.data.get(HostType.managed)
        if not managed_contexts:
            managed_contexts = create_managed_contexts(control_contexts)
        control_type = 'posix'
        if isinstance(args, WindowsIntegrationConfig):
            managed_type = 'windows'
        else:
            managed_type = 'posix'
        control_state: dict[str, tuple[list[str], list[SshProcess]]] = {}
        managed_state: dict[str, tuple[list[str], list[SshProcess]]] = {}

        def pre_target(target: IntegrationTarget) -> None:
            """Configure hosts for SSH port forwarding required by the specified target."""
            forward_ssh_ports(args, control_connections, '%s_hosts_prepare.yml' % control_type, control_state, target, HostType.control, control_contexts)
            forward_ssh_ports(args, managed_connections, '%s_hosts_prepare.yml' % managed_type, managed_state, target, HostType.managed, managed_contexts)

        def post_target(target: IntegrationTarget) -> None:
            """Clean up previously configured SSH port forwarding which was required by the specified target."""
            cleanup_ssh_ports(args, control_connections, '%s_hosts_restore.yml' % control_type, control_state, target, HostType.control)
            cleanup_ssh_ports(args, managed_connections, '%s_hosts_restore.yml' % managed_type, managed_state, target, HostType.managed)
    else:
        pre_target, post_target = (None, None)
    return (pre_target, post_target)