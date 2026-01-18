from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def convert_logging(value: str | int) -> PyLogLevel:
    """Convert a string to a Python logging level

    If a log level is passed in, it is returned as-is. Otherwise the function
    understands the following strings, ignoring case:

    * "critical"
    * "error"
    * "warning"
    * "info"
    * "debug"
    * "trace"
    * "none"

    Args:
        value (str):
            A string value to convert to a logging level

    Returns:
        int or None

    Raises:
        ValueError

    """
    if value is None or isinstance(value, int):
        if value in set(_log_levels.values()):
            return value
    else:
        value = value.upper()
        if value in _log_levels:
            return _log_levels[value]
    raise ValueError(f'Cannot convert {value} to log level, valid values are: {', '.join(_log_levels)}')