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
def copy2_safe(src, dst, log=None):
    """copy src to dst

    like shutil.copy2, but log errors in copystat instead of raising
    """
    shutil.copyfile(src, dst)
    try:
        shutil.copystat(src, dst)
    except OSError:
        if log:
            log.debug('copystat on %s failed', dst, exc_info=True)