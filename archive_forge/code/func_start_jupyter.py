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
def start_jupyter():
    global JUPYTER_PORT, JUPYTER_PROCESS
    args = ['jupyter', 'server', '--port', str(JUPYTER_PORT), "--NotebookApp.token=''"]
    JUPYTER_PROCESS = process = Popen(args, stdout=PIPE, stderr=PIPE, bufsize=1, encoding='utf-8')
    deadline = time.monotonic() + JUPYTER_TIMEOUT
    while True:
        line = process.stderr.readline()
        time.sleep(0.02)
        if 'http://127.0.0.1:' in line:
            host = 'http://127.0.0.1:'
            break
        if 'http://localhost:' in line:
            host = 'http://localhost:'
            break
        if time.monotonic() > deadline:
            raise TimeoutError('jupyter server did not start within {timeout} seconds.')
    JUPYTER_PORT = int(line.split(host)[-1][:4])