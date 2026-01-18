from __future__ import annotations
import typing
from collections import OrderedDict
from enum import Enum, auto
from threading import RLock
def ensure_can_construct_http_header_dict(potential: object) -> ValidHTTPHeaderSource | None:
    if isinstance(potential, HTTPHeaderDict):
        return potential
    elif isinstance(potential, typing.Mapping):
        return typing.cast(typing.Mapping[str, str], potential)
    elif isinstance(potential, typing.Iterable):
        return typing.cast(typing.Iterable[typing.Tuple[str, str]], potential)
    elif hasattr(potential, 'keys') and hasattr(potential, '__getitem__'):
        return typing.cast('HasGettableStringKeys', potential)
    else:
        return None