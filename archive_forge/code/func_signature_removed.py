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
@contextmanager
def signature_removed(nb):
    """Context manager for operating on a notebook with its signature removed

    Used for excluding the previous signature when computing a notebook's signature.
    """
    save_signature = nb['metadata'].pop('signature', None)
    try:
        yield
    finally:
        if save_signature is not None:
            nb['metadata']['signature'] = save_signature