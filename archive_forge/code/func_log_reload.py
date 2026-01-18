from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def log_reload(self, filename: str) -> None:
    filename = os.path.abspath(filename)
    _log('info', f' * Detected change in {filename!r}, reloading')