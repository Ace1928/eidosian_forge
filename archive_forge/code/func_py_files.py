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
@pytest.fixture
def py_files():
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    tf2 = tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=os.path.split(tf.name)[0], delete=False)
    try:
        yield (tf, tf2)
    finally:
        tf.close()
        tf2.close()
        os.unlink(tf.name)
        os.unlink(tf2.name)