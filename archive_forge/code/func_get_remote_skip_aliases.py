from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
def get_remote_skip_aliases(config: RemoteConfig) -> dict[str, str]:
    """Return a dictionary of skip aliases and the reason why they apply."""
    return get_platform_skip_aliases(config.platform, config.version, config.arch)