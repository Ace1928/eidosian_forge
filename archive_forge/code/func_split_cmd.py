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
def split_cmd(cmd: str) -> List[str]:
    import platform
    posix = platform.uname().system.lower() != 'windows'
    return shlex.split(cmd, posix=posix)