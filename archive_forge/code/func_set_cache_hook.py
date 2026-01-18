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
def set_cache_hook(hook: Callable[[CustomModel, Implementation], AttrDict | None]) -> None:
    """Sets a compiled model cache hook used to look up the compiled
       code given the CustomModel and Implementation"""
    global _CACHING_IMPLEMENTATION
    _CACHING_IMPLEMENTATION = hook