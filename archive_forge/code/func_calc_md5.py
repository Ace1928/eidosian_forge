from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def calc_md5(s: bytes | str) -> str:
    """Return the md5 hash of the given string."""
    h = hashlib.new('md5', **HASHLIB_KWARGS)
    b = s.encode('utf-8') if isinstance(s, str) else s
    h.update(b)
    return h.hexdigest()