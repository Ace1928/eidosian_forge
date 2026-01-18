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
def colorize_console() -> bool:
    _colorize_console: bool = getattr(sys.stdout, 'colorize_console', None)
    if _colorize_console is not None:
        return _colorize_console
    try:
        if is_windows():
            _colorize_console = os.isatty(sys.stdout.fileno()) and _windows_ansi()
        else:
            _colorize_console = os.isatty(sys.stdout.fileno()) and os.environ.get('TERM', 'dumb') != 'dumb'
    except Exception:
        _colorize_console = False
    sys.stdout.colorize_console = _colorize_console
    return _colorize_console