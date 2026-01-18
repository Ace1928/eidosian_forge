from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def is_dev() -> bool:
    return convert_bool(os.environ.get('BOKEH_DEV', False))