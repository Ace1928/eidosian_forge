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
def bundle_models(models: Sequence[type[HasProps]] | None) -> str | None:
    """Create a bundle of selected `models`. """
    custom_models = _get_custom_models(models)
    if custom_models is None:
        return None
    key = calc_cache_key(custom_models)
    bundle = _bundle_cache.get(key, None)
    if bundle is None:
        try:
            _bundle_cache[key] = bundle = _bundle_models(custom_models)
        except CompilationError as error:
            print('Compilation failed:', file=sys.stderr)
            print(str(error), file=sys.stderr)
            sys.exit(1)
    return bundle