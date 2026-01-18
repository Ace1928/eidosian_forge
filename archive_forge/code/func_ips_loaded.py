from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def ips_loaded(*args: Any, **kwargs: Any) -> Any:
    _load_ips()
    return f(*args, **kwargs)