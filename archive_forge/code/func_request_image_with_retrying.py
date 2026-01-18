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
@tenacity.retry(wait=tenacity.wait_random(min=5, max=10), stop=tenacity.stop_after_delay(60000))
def request_image_with_retrying(**kwargs):
    """
    Helper method to perform an image request to a running orca server process
    with retrying logic.
    """
    from requests import post
    from plotly.io.json import to_json_plotly
    if config.server_url:
        server_url = config.server_url
    else:
        server_url = 'http://{hostname}:{port}'.format(hostname='localhost', port=orca_state['port'])
    request_params = {k: v for k, v in kwargs.items() if v is not None}
    json_str = to_json_plotly(request_params)
    response = post(server_url + '/', data=json_str)
    if response.status_code == 522:
        shutdown_server()
        ensure_server()
        raise OSError('522: client socket timeout')
    return response