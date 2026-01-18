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
def shutdown_server():
    """
    Shutdown the running orca server process, if any

    Returns
    -------
    None
    """
    if orca_state['proc'] is not None:
        with orca_lock:
            if orca_state['proc'] is not None:
                parent = psutil.Process(orca_state['proc'].pid)
                for child in parent.children(recursive=True):
                    try:
                        child.terminate()
                    except:
                        pass
                try:
                    orca_state['proc'].terminate()
                    child_status = orca_state['proc'].wait()
                except:
                    pass
                orca_state['proc'] = None
                if orca_state['shutdown_timer'] is not None:
                    orca_state['shutdown_timer'].cancel()
                    orca_state['shutdown_timer'] = None
                orca_state['port'] = None
                status._props['state'] = 'validated'
                status._props['pid'] = None
                status._props['port'] = None
                status._props['command'] = None