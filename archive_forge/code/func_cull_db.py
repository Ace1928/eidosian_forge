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
def cull_db(self):
    """Cull oldest 25% of the trusted signatures when the size limit is reached"""
    self.db.execute('DELETE FROM nbsignatures WHERE id IN (\n            SELECT id FROM nbsignatures ORDER BY last_seen DESC LIMIT -1 OFFSET ?\n        );\n        ', (max(int(0.75 * self.cache_size), 1),))