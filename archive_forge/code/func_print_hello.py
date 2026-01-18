from __future__ import annotations
import typing as T
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import noKwargs, noPosargs
@noKwargs
@noPosargs
def print_hello(self, state: ModuleState, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> None:
    print('Hello from a Meson module')