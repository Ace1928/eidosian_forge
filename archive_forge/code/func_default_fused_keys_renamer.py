from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def default_fused_keys_renamer(keys, max_fused_key_length=120):
    """Create new keys for ``fuse`` tasks.

    The optional parameter `max_fused_key_length` is used to limit the maximum string length for each renamed key.
    If this parameter is set to `None`, there is no limit.
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)
    if max_fused_key_length:
        max_fused_key_length -= 5

    def _enforce_max_key_limit(key_name):
        if max_fused_key_length and len(key_name) > max_fused_key_length:
            name_hash = f'{hash(key_name):x}'[:4]
            key_name = f'{key_name[:max_fused_key_length]}-{name_hash}'
        return key_name
    if typ is str:
        first_name = utils.key_split(first_key)
        names = {utils.key_split(k) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key)
        concatenated_name = '-'.join(names)
        return _enforce_max_key_limit(concatenated_name)
    elif typ is tuple and len(first_key) > 0 and isinstance(first_key[0], str):
        first_name = utils.key_split(first_key)
        names = {utils.key_split(k) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key[0])
        concatenated_name = '-'.join(names)
        return (_enforce_max_key_limit(concatenated_name),) + first_key[1:]