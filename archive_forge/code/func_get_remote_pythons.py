from __future__ import annotations
from ...constants import (
from ...completion import (
from ...host_configs import (
def get_remote_pythons(name: str, controller: bool, strict: bool) -> list[str]:
    """Return a list of remote instance Python versions supported by the specified host config."""
    platform_config = filter_completion(remote_completion()).get(name)
    available_pythons = CONTROLLER_PYTHON_VERSIONS if controller else SUPPORTED_PYTHON_VERSIONS
    if not platform_config:
        return [] if strict else list(available_pythons)
    supported_pythons = [python for python in platform_config.supported_pythons if python in available_pythons]
    return supported_pythons