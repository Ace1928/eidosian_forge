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
def new_untitled(self, path='', type='', ext=''):
    """Create a new untitled file or directory in path

        path must be a directory

        File extension can be specified.

        Use `new` to create files with a fully specified path (including filename).
        """
    path = path.strip('/')
    if not self.dir_exists(path):
        raise HTTPError(404, 'No such directory: %s' % path)
    model = {}
    if type:
        model['type'] = type
    if ext == '.ipynb':
        model.setdefault('type', 'notebook')
    else:
        model.setdefault('type', 'file')
    insert = ''
    if model['type'] == 'directory':
        untitled = self.untitled_directory
        insert = ' '
    elif model['type'] == 'notebook':
        untitled = self.untitled_notebook
        ext = '.ipynb'
    elif model['type'] == 'file':
        untitled = self.untitled_file
    else:
        raise HTTPError(400, 'Unexpected model type: %r' % model['type'])
    name = self.increment_filename(untitled + ext, path, insert=insert)
    path = f'{path}/{name}'
    return self.new(model, path)