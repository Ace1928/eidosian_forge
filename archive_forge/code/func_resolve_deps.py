from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
def resolve_deps(deps: list[str], root: str) -> dict[str, str]:
    custom_modules = {model.module for model in custom_models.values()}
    missing = set(deps) - known_modules - custom_modules
    return resolve_modules(missing, root)