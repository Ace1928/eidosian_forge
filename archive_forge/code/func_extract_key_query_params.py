from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def extract_key_query_params(query_params: dict[str, list[str]], param_key: str) -> set[str]:
    """Extracts key (case-insensitive) query params from Dict, and returns them as Set of str."""
    return {item.lower() for sublist in [[value.lower() for value in query_params[key]] for key in query_params.keys() if key.lower() == param_key and query_params.get(key)] for item in sublist}