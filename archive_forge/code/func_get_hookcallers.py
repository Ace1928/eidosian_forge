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
def get_hookcallers(self, plugin: _Plugin) -> list[HookCaller] | None:
    """Get all hook callers for the specified plugin.

        :returns:
            The hook callers, or ``None`` if ``plugin`` is not registered in
            this plugin manager.
        """
    if self.get_name(plugin) is None:
        return None
    hookcallers = []
    for hookcaller in self.hook.__dict__.values():
        for hookimpl in hookcaller.get_hookimpls():
            if hookimpl.plugin is plugin:
                hookcallers.append(hookcaller)
    return hookcallers