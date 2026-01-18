from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def convert_ico_path(value: str) -> str:
    """Convert a string to an .ico path

    Args:
        value (str):
            A string value to convert to a .ico path

    Returns:
        string

    Raises:
        ValueError

    """
    lowered = value.lower()
    if lowered == 'none':
        return 'none'
    if lowered == 'default':
        return str(server_path() / 'views' / 'bokeh.ico')
    if lowered == 'default-dev':
        return str(server_path() / 'views' / 'bokeh-dev.ico')
    if not value.endswith('.ico'):
        raise ValueError(f'Cannot convert {value!r} to valid .ico path')
    return value