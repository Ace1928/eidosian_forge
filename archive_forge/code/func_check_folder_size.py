from __future__ import annotations
import errno
import math
import mimetypes
import os
import platform
import shutil
import stat
import subprocess
import sys
import typing as t
import warnings
from datetime import datetime
from pathlib import Path
import nbformat
from anyio.to_thread import run_sync
from jupyter_core.paths import exists, is_file_hidden, is_hidden
from send2trash import send2trash
from tornado import web
from traitlets import Bool, Int, TraitError, Unicode, default, validate
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import AuthenticatedFileHandler
from jupyter_server.transutils import _i18n
from jupyter_server.utils import to_api_path
from .filecheckpoints import AsyncFileCheckpoints, FileCheckpoints
from .fileio import AsyncFileManagerMixin, FileManagerMixin
from .manager import AsyncContentsManager, ContentsManager, copy_pat
def check_folder_size(self, path):
    """
        limit the size of folders being copied to be no more than the
        trait max_copy_folder_size_mb to prevent a timeout error
        """
    limit_bytes = self.max_copy_folder_size_mb * 1024 * 1024
    size = int(self._get_dir_size(self._get_os_path(path)))
    size = size * 1024 if platform.system() == 'Darwin' else size
    if size > limit_bytes:
        raise web.HTTPError(400, f'''\n                    Can't copy folders larger than {self.max_copy_folder_size_mb}MB,\n                    "{path}" is {self._human_readable_size(size)}\n                ''')