from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
@property
def config_system(self) -> dict[str, Any]:
    return dict(self._config_system)