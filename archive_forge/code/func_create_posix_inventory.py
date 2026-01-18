from __future__ import annotations
import shutil
import typing as t
from .config import (
from .util import (
from .host_profiles import (
from .ssh import (
def create_posix_inventory(args: EnvironmentConfig, path: str, target_hosts: list[HostProfile], needs_ssh: bool=False) -> None:
    """Create and return inventory for use in POSIX integration tests."""
    target_hosts = t.cast(list[SshTargetHostProfile], target_hosts)
    if len(target_hosts) != 1:
        raise Exception()
    target_host = target_hosts[0]
    if isinstance(target_host, ControllerProfile) and (not needs_ssh):
        inventory = Inventory(host_groups=dict(testgroup=dict(testhost=dict(ansible_connection='local', ansible_pipelining='yes', ansible_python_interpreter=target_host.python.path))))
    else:
        connections = target_host.get_controller_target_connections()
        if len(connections) != 1:
            raise Exception()
        ssh = connections[0]
        testhost: dict[str, t.Optional[t.Union[str, int]]] = dict(ansible_connection='ssh', ansible_pipelining='yes', ansible_python_interpreter=ssh.settings.python_interpreter, ansible_host=ssh.settings.host, ansible_port=ssh.settings.port, ansible_user=ssh.settings.user, ansible_ssh_private_key_file=ssh.settings.identity_file, ansible_ssh_extra_args=ssh_options_to_str(ssh.settings.options))
        if ssh.become:
            testhost.update(ansible_become='yes', ansible_become_method=ssh.become.method)
        testhost = exclude_none_values(testhost)
        inventory = Inventory(host_groups=dict(testgroup=dict(testhost=testhost)))
    inventory.write(args, path)