import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def launch_browser(attempt_launch_browser: bool=True) -> bool:
    """Decide if we should launch a browser."""
    _display_variables = ['DISPLAY', 'WAYLAND_DISPLAY', 'MIR_SOCKET']
    _webbrowser_names_blocklist = ['www-browser', 'lynx', 'links', 'elinks', 'w3m']
    import webbrowser
    launch_browser = attempt_launch_browser
    if launch_browser:
        if 'linux' in sys.platform and (not any((os.getenv(var) for var in _display_variables))):
            launch_browser = False
        try:
            browser = webbrowser.get()
            if hasattr(browser, 'name') and browser.name in _webbrowser_names_blocklist:
                launch_browser = False
        except webbrowser.Error:
            launch_browser = False
    return launch_browser