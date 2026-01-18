from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from ..util import parseBoolValue
def getConfigs(self) -> dict[str, Any]:
    """
        Return all configuration options.

        Returns:
            All configuration options.
        """
    return {key: self.getConfig(key) for key in self.config.keys()}