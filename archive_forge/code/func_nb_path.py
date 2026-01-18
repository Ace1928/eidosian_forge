from the server to the kernel.
from __future__ import annotations
import asyncio
import calendar
import datetime as dt
import inspect
import json
import logging
import os
import pathlib
import textwrap
import time
from queue import Empty
from typing import Any, Awaitable
from urllib.parse import urljoin
import tornado
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.exceptions import ProtocolError
from bokeh.protocol.receiver import Receiver
from bokeh.server.tornado import DEFAULT_KEEP_ALIVE_MS
from bokeh.server.views.multi_root_static_handler import MultiRootStaticHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import (
from jupyter_server.base.handlers import JupyterHandler
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..config import config
from .resources import DIST_DIR, ERROR_TEMPLATE, _env
from .server import COMPONENT_PATH, ComponentResourceHandler
from .state import state
import os
import pathlib
import sys
from panel.io.jupyter_executor import PanelExecutor
def nb_path(self, path=None):
    root_dir = get_server_root_dir(self.application.settings)
    rel_path = pathlib.Path(self.notebook_path or path)
    if rel_path.is_absolute():
        notebook_path = str(rel_path)
    else:
        notebook_path = str((root_dir / rel_path).absolute())
    return pathlib.Path(notebook_path)