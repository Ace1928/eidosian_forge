from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
@cache
def get_host_target_type_map() -> dict[t.Type[HostConfig], t.Type[TargetFilter]]:
    """Create and return a mapping of HostConfig types to TargetFilter types."""
    return get_type_map(TargetFilter, HostConfig)