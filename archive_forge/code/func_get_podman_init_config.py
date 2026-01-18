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
def get_podman_init_config(self) -> InitConfig:
    """Return init config for running under Podman."""
    options = self.get_common_run_options()
    command: t.Optional[str] = None
    command_privileged = False
    expected_mounts: tuple[CGroupMount, ...]
    cgroup_version = get_docker_info(self.args).cgroup_version
    options.extend(('--cap-add', 'SYS_CHROOT'))
    if self.config.audit == AuditMode.REQUIRED and detect_host_properties(self.args).audit_code == 'EPERM':
        options.extend(('--cap-add', 'AUDIT_WRITE'))
    if (loginuid := detect_host_properties(self.args).loginuid) not in (0, LOGINUID_NOT_SET, None):
        display.warning(f'Running containers with capability AUDIT_CONTROL since the container loginuid ({loginuid}) is incorrect. This is most likely due to use of sudo to run ansible-test when loginuid is already set.', unique=True)
        options.extend(('--cap-add', 'AUDIT_CONTROL'))
    if self.config.cgroup == CGroupVersion.NONE:
        options.extend(('--systemd', 'false', '--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
    elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
        options.extend(('--systemd', 'always', '--cgroupns', 'host', '--tmpfs', '/sys/fs/cgroup'))
        self.check_systemd_cgroup_v1(options)
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=None, state=CGroupState.HOST), CGroupMount(path=CGroupPath.SYSTEMD_RELEASE_AGENT, type=None, writable=False, state=None))
    elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
        options.extend(('--systemd', 'always', '--cgroupns', 'private'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
    elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
        cgroup_path = self.create_systemd_cgroup_v1()
        command = f'echo 1 > {cgroup_path}/cgroup.procs'
        options.extend(('--systemd', 'always', '--cgroupns', 'private', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:ro', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=False, state=CGroupState.SHADOWED), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
    else:
        raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
    return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)