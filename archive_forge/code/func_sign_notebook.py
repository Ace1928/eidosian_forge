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
def sign_notebook(self, nb, notebook_path='<stdin>'):
    """Sign a notebook that's been loaded"""
    if self.notary.check_signature(nb):
        print('Notebook already signed: %s' % notebook_path)
    else:
        print('Signing notebook: %s' % notebook_path)
        self.notary.sign(nb)