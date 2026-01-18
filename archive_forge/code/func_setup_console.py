from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
def setup_console() -> None:
    if is_windows():
        try:
            delattr(sys.stdout, 'colorize_console')
        except AttributeError:
            pass