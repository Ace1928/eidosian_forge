import asyncio
import atexit
import os
import pathlib
import re
import shutil
import signal
import socket
import tempfile
import time
import unittest
from contextlib import contextmanager
from subprocess import PIPE, Popen
import pandas as pd
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc as set_bkdoc
from pyviz_comms import Comm
from panel import config, serve
from panel.config import panel_extension
from panel.io.reload import (
from panel.io.state import set_curdoc, state
from panel.pane import HTML, Markdown
@pytest.fixture(autouse=True)
def module_cleanup():
    """
    Cleanup Panel extensions after each test.
    """
    from bokeh.core.has_props import _default_resolver
    to_reset = list(panel_extension._imports.values())
    _default_resolver._known_models = {name: model for name, model in _default_resolver._known_models.items() if not any((model.__module__.startswith(tr) for tr in to_reset))}