import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
def ensure_server():
    """
    Start an orca server if none is running. If a server is already running,
    then reset the timeout countdown

    Returns
    -------
    None
    """
    if psutil is None:
        raise ValueError('Image generation requires the psutil package.\n\nInstall using pip:\n    $ pip install psutil\n\nInstall using conda:\n    $ conda install psutil\n')
    if not get_module('requests'):
        raise ValueError('Image generation requires the requests package.\n\nInstall using pip:\n    $ pip install requests\n\nInstall using conda:\n    $ conda install requests\n')
    if not config.server_url:
        if status.state == 'unvalidated':
            validate_executable()
        with orca_lock:
            if orca_state['shutdown_timer'] is not None:
                orca_state['shutdown_timer'].cancel()
            if orca_state['proc'] is None:
                if config.port is None:
                    orca_state['port'] = find_open_port()
                else:
                    orca_state['port'] = config.port
                cmd_list = status._props['executable_list'] + ['serve', '-p', str(orca_state['port']), '--plotly', config.plotlyjs, '--graph-only']
                if config.topojson:
                    cmd_list.extend(['--topojson', config.topojson])
                if config.mathjax:
                    cmd_list.extend(['--mathjax', config.mathjax])
                if config.mapbox_access_token:
                    cmd_list.extend(['--mapbox-access-token', config.mapbox_access_token])
                DEVNULL = open(os.devnull, 'wb')
                with orca_env():
                    stderr = DEVNULL if 'CI' in os.environ else None
                    orca_state['proc'] = subprocess.Popen(cmd_list, stdout=DEVNULL, stderr=stderr)
                status._props['state'] = 'running'
                status._props['pid'] = orca_state['proc'].pid
                status._props['port'] = orca_state['port']
                status._props['command'] = cmd_list
            if config.timeout is not None:
                t = threading.Timer(config.timeout, shutdown_server)
                t.daemon = True
                t.start()
                orca_state['shutdown_timer'] = t