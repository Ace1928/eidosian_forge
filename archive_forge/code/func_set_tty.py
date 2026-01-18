import contextlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from importlib import import_module
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import anyio
from .filters import DefaultFilter
from .main import Change, FileChange, awatch, watch
@contextlib.contextmanager
def set_tty(tty_path: Optional[str]) -> Generator[None, None, None]:
    if tty_path:
        try:
            with open(tty_path) as tty:
                sys.stdin = tty
                yield
        except OSError:
            yield
    else:
        yield