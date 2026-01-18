import asyncio
import logging
import os
import signal
import sys
import traceback
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Iterable, Optional, Set
from mypy_extensions import mypyc_attr
from black import WriteBack, format_file_in_place
from black.cache import Cache
from black.mode import Mode
from black.output import err
from black.report import Changed, Report
def maybe_install_uvloop() -> None:
    """If our environment has uvloop installed we use it.

    This is called only from command-line entry points to avoid
    interfering with the parent process if Black is used as a library.
    """
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass