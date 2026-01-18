import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
@contextlib.contextmanager
def run_panel_serve(args, cwd=None):
    cmd = [sys.executable, '-m', 'panel', 'serve'] + args
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, cwd=cwd, close_fds=ON_POSIX)
    try:
        yield p
    except Exception as e:
        p.terminate()
        p.wait()
        print('An error occurred: %s', e)
        try:
            out = p.stdout.read().decode()
            print('\n---- subprocess stdout follows ----\n')
            print(out)
        except Exception:
            pass
        raise
    else:
        p.terminate()
        p.wait()