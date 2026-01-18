from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from ..util import parseBoolValue
def setConfigs(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
    """
        Loop through a collection of configuration options, passing each to
        [`setConfig`][markdown.extensions.Extension.setConfig].

        Arguments:
            items: Collection of configuration options.

        Raises:
            KeyError: for any unknown key.
        """
    if hasattr(items, 'items'):
        items = items.items()
    for key, value in items:
        self.setConfig(key, value)