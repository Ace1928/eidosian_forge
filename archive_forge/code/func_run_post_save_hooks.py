from __future__ import annotations
import itertools
import json
import os
import re
import typing as t
import warnings
from fnmatch import fnmatch
from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item
from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints
def run_post_save_hooks(self, model, os_path):
    """Run the post-save hooks if any, and log errors"""
    post_save_hooks = [self.post_save_hook] if self.post_save_hook is not None else []
    post_save_hooks += self._post_save_hooks
    for post_save_hook in post_save_hooks:
        try:
            self.log.debug('Running post-save hook on %s', os_path)
            post_save_hook(os_path=os_path, model=model, contents_manager=self)
        except Exception as e:
            self.log.error('Post-save %s hook failed on %s', post_save_hook.__name__, os_path, exc_info=True)
            raise HTTPError(500, 'Unexpected error while running post hook save: %s' % e) from e