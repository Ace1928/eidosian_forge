from __future__ import annotations
import sys
from functools import partial
from typing import Any, Callable, Tuple, Type, cast
from attrs import fields, has, resolve_types
from cattrs import Converter
from cattrs.gen import (
from fontTools.misc.transform import Transform
def custom_structure_hook_factory(cls: Type[Any]) -> Callable[[Any, Any], Any]:
    return partial(cls._structure, converter=conv)