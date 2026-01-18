from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from .registry import _ET
from .registry import _ListenerFnType
from .. import util
from ..util.compat import FullArgSpec
def leg(fn: Callable[..., Any]) -> Callable[..., Any]:
    if not hasattr(fn, '_legacy_signatures'):
        fn._legacy_signatures = []
    fn._legacy_signatures.append((since, argnames, converter))
    return fn