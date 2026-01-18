from __future__ import annotations
import errno
import hashlib
import os
import shutil
from base64 import decodebytes, encodebytes
from contextlib import contextmanager
from functools import partial
import nbformat
from anyio.to_thread import run_sync
from tornado.web import HTTPError
from traitlets import Bool, Enum
from traitlets.config import Configurable
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.utils import ApiPath, to_api_path, to_os_path
@contextmanager
def atomic_writing(self, os_path, *args, **kwargs):
    """wrapper around atomic_writing that turns permission errors to 403.
        Depending on flag 'use_atomic_writing', the wrapper perform an actual atomic writing or
        simply writes the file (whatever an old exists or not)"""
    with self.perm_to_403(os_path):
        kwargs['log'] = self.log
        if self.use_atomic_writing:
            with atomic_writing(os_path, *args, **kwargs) as f:
                yield f
        else:
            with _simple_writing(os_path, *args, **kwargs) as f:
                yield f