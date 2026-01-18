from __future__ import annotations
import collections.abc
import uuid
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from functools import lru_cache
from typing import (
import attrs
from attrs import define, field
from fontTools.misc.arrayTools import unionRect
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen, ControlBoundsPen
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import OBJECT_LIBS_KEY
from ufoLib2.typing import Drawable, GlyphSet, HasIdentifier
@classmethod
def coerce_from_dict(cls: Type[_T], value: _T | Mapping[str, Any]) -> _T:
    if isinstance(value, cls):
        return value
    elif isinstance(value, Mapping):
        attr_map = cls._key_to_attr_map()
        return cls(**{attr_map[k]: v for k, v in value.items()})
    raise TypeError(f'Expected {cls.__name__} or mapping, found: {type(value).__name__}')