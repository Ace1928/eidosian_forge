from __future__ import annotations
import inspect
import types
import warnings
from typing import Any
from typing import Callable
from typing import cast
from typing import Final
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING
from . import _tracing
from ._callers import _multicall
from ._hooks import _HookImplFunction
from ._hooks import _Namespace
from ._hooks import _Plugin
from ._hooks import _SubsetHookCaller
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplOpts
from ._hooks import HookRelay
from ._hooks import HookspecOpts
from ._hooks import normalize_hookimpl_opts
from ._result import Result
def load_setuptools_entrypoints(self, group: str, name: str | None=None) -> int:
    """Load modules from querying the specified setuptools ``group``.

        :param group:
            Entry point group to load plugins.
        :param name:
            If given, loads only plugins with the given ``name``.

        :return:
            The number of plugins loaded by this call.
        """
    import importlib.metadata
    count = 0
    for dist in list(importlib.metadata.distributions()):
        for ep in dist.entry_points:
            if ep.group != group or (name is not None and ep.name != name) or self.get_plugin(ep.name) or self.is_blocked(ep.name):
                continue
            plugin = ep.load()
            self.register(plugin, name=ep.name)
            self._plugin_distinfo.append((plugin, DistFacade(dist)))
            count += 1
    return count