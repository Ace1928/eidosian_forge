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
def catch_sigterm() -> None:
    """
    Catch SIGTERM and raise KeyboardInterrupt instead. This means watchfiles will stop quickly
    on `docker compose stop` and other cases where SIGTERM is sent.

    Without this the watchfiles process will be killed while a running process will continue uninterrupted.
    """
    logger.debug('registering handler for SIGTERM on watchfiles process %d', os.getpid())
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)