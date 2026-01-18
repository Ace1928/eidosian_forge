from __future__ import annotations
import base64
import dataclasses
import hashlib
import inspect
import operator
import platform
import sys
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
def local_dataclass_fields(cls: Type[Any]) -> Iterable[dataclasses.Field[Any]]:
    """Return a sequence of all dataclasses.Field objects associated with
    an already processed dataclass, excluding those that originate from a
    superclass.

    The class must **already be a dataclass** for Field objects to be returned.

    """
    if dataclasses.is_dataclass(cls):
        super_fields: Set[dataclasses.Field[Any]] = set()
        for sup in cls.__bases__:
            super_fields.update(dataclass_fields(sup))
        return [f for f in dataclasses.fields(cls) if f not in super_fields]
    else:
        return []