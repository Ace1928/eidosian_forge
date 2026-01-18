from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from ..util import parseBoolValue
def getConfigInfo(self) -> list[tuple[str, str]]:
    """
        Return descriptions of all configuration options.

        Returns:
            All descriptions of configuration options.
        """
    return [(key, self.config[key][1]) for key in self.config.keys()]