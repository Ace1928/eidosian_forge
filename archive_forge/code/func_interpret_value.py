from __future__ import annotations
import ast
import base64
import builtins  # Explicitly use builtins.set as 'set' will be shadowed by a function
import json
import os
import pathlib
import site
import sys
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, overload
import yaml
from dask.typing import no_default
def interpret_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    hardcoded_map = {'none': None, 'null': None, 'false': False, 'true': True}
    return hardcoded_map.get(value.lower(), value)