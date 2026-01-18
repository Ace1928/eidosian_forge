from __future__ import annotations
import importlib.metadata as importlib_metadata
import warnings
from typing import TYPE_CHECKING, Final, Iterable
def get_plugins() -> Iterable[PydanticPluginProtocol]:
    """Load plugins for Pydantic.

    Inspired by: https://github.com/pytest-dev/pluggy/blob/1.3.0/src/pluggy/_manager.py#L376-L402
    """
    global _plugins, _loading_plugins
    if _loading_plugins:
        return ()
    elif _plugins is None:
        _plugins = {}
        _loading_plugins = True
        try:
            for dist in importlib_metadata.distributions():
                for entry_point in dist.entry_points:
                    if entry_point.group != PYDANTIC_ENTRY_POINT_GROUP:
                        continue
                    if entry_point.value in _plugins:
                        continue
                    try:
                        _plugins[entry_point.value] = entry_point.load()
                    except (ImportError, AttributeError) as e:
                        warnings.warn(f'{e.__class__.__name__} while loading the `{entry_point.name}` Pydantic plugin, this plugin will not be installed.\n\n{e!r}')
        finally:
            _loading_plugins = False
    return _plugins.values()