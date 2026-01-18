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
def env_var(self) -> str | None:
    return self._env_var