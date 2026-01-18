from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def lower_clean_dict_keys(dict: Mapping[_Key, _Value]) -> dict[str, _Value]:
    return {k.lower().strip(): v for k, v in dict.items()}