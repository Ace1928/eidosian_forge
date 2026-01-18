import os
import shutil
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode
from jupyter_server import _tz as tz
from .checkpoints import (
from .fileio import AsyncFileManagerMixin, FileManagerMixin
def no_such_checkpoint(self, path, checkpoint_id):
    raise HTTPError(404, f'Checkpoint does not exist: {path}@{checkpoint_id}')