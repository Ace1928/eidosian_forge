from __future__ import annotations
import sys
from functools import partial
from typing import Any, Callable, Tuple, Type, cast
from attrs import fields, has, resolve_types
from cattrs import Converter
from cattrs.gen import (
from fontTools.misc.transform import Transform
def is_ufoLib2_attrs_class(cls: Type[Any]) -> bool:
    return is_ufoLib2_class(cls) and (has(cls) or has(get_origin(cls)))