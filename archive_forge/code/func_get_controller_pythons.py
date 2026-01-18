from __future__ import annotations
from ...constants import (
from ...completion import (
from ...host_configs import (
def get_controller_pythons(controller_config: HostConfig, strict: bool) -> list[str]:
    """Return a list of controller Python versions supported by the specified host config."""
    if isinstance(controller_config, DockerConfig):
        pythons = get_docker_pythons(controller_config.name, False, strict)
    elif isinstance(controller_config, PosixRemoteConfig):
        pythons = get_remote_pythons(controller_config.name, False, strict)
    else:
        pythons = list(SUPPORTED_PYTHON_VERSIONS)
    return pythons