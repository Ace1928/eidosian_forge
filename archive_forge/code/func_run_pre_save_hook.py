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
def run_pre_save_hook(self, model, path, **kwargs):
    """Run the pre-save hook if defined, and log errors"""
    warnings.warn('run_pre_save_hook is deprecated, use run_pre_save_hooks instead.', DeprecationWarning, stacklevel=2)
    if self.pre_save_hook:
        try:
            self.log.debug('Running pre-save hook on %s', path)
            self.pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
        except HTTPError:
            raise
        except Exception:
            self.log.error('Pre-save hook failed on %s', path, exc_info=True)