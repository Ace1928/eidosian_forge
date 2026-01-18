from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def exclude_keys_in_dict(d: dict[str, Any], keys_to_exclude: list[str]) -> dict[str, Any]:
    """Returns new object but without keys defined in keys_to_exclude"""
    return {key: value for key, value in d.items() if key.lower() not in keys_to_exclude}