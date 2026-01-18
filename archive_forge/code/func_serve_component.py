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
def serve_component(page, app, suffix='', wait=True, **kwargs):
    msgs = []
    page.on('console', lambda msg: msgs.append(msg))
    port = serve_and_wait(app, page, **kwargs)
    page.goto(f'http://localhost:{port}{suffix}')
    if wait:
        wait_until(lambda: any(('Websocket connection 0 is now open' in str(msg) for msg in msgs)), page, interval=10)
    return (msgs, port)