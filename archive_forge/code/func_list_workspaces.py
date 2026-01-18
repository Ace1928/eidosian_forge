from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import urllib
from pathlib import Path
from typing import Any
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from traitlets.config import LoggingConfigurable
def list_workspaces(self) -> list:
    """List all available workspaces."""
    prefix = slugify('', sign=False)
    return _list_workspaces(self.workspaces_dir, prefix)