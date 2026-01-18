from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
def sign_notebook_file(self, notebook_path):
    """Sign a notebook from the filesystem"""
    if not Path(notebook_path).exists():
        self.log.error('Notebook missing: %s', notebook_path)
        self.exit(1)
    with Path(notebook_path).open(encoding='utf8') as f:
        nb = read(f, NO_CONVERT)
    self.sign_notebook(nb, notebook_path)