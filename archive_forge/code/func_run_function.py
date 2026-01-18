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
def run_function(function: str, tty_path: Optional[str], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    with set_tty(tty_path):
        func = import_string(function)
        func(*args, **kwargs)