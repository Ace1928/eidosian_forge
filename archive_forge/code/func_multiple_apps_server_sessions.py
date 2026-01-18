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
def multiple_apps_server_sessions(port):
    """Serve multiple apps and yield a factory to allow
    parameterizing the slugs and the titles."""
    servers = []

    def create_sessions(slugs, titles):
        app1_slug, app2_slug = slugs
        apps = {app1_slug: Markdown('First app'), app2_slug: Markdown('Second app')}
        server = serve(apps, port=port, title=titles, show=False, start=False)
        servers.append(server)
        session1 = pull_session(url=f'http://localhost:{server.port:d}/app1', io_loop=server.io_loop)
        session2 = pull_session(url=f'http://localhost:{server.port:d}/app2', io_loop=server.io_loop)
        return (session1, session2)
    yield create_sessions
    for server in servers:
        try:
            server.stop()
        except AssertionError:
            continue