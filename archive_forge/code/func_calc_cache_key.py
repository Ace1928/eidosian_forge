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
def calc_cache_key(custom_models: dict[str, CustomModel]) -> str:
    """ Generate a key to cache a custom extension implementation with.

    There is no metadata other than the Model classes, so this is the only
    base to generate a cache key.

    We build the model keys from the list of ``model.full_name``. This is
    not ideal but possibly a better solution can be found found later.

    """
    model_names = {model.full_name for model in custom_models.values()}
    encoded_names = ','.join(sorted(model_names)).encode('utf-8')
    return hashlib.sha256(encoded_names).hexdigest()