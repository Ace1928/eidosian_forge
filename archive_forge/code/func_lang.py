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
@property
def lang(self) -> str:
    if self.file is not None:
        if self.file.endswith('.ts'):
            return 'typescript'
        if self.file.endswith('.js'):
            return 'javascript'
        if self.file.endswith(('.css', '.less')):
            return 'less'
    raise ValueError(f'unknown file type {self.file}')