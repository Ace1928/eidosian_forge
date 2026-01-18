from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
@classmethod
def check_unsafe_options(cls, options: List[str], unsafe_options: List[str]) -> None:
    """Check for unsafe options.

        Some options that are passed to `git <command>` can be used to execute
        arbitrary commands, this are blocked by default.
        """
    bare_unsafe_options = [option.lstrip('-') for option in unsafe_options]
    for option in options:
        for unsafe_option, bare_option in zip(unsafe_options, bare_unsafe_options):
            if option.startswith(unsafe_option) or option == bare_option:
                raise UnsafeOptionError(f'{unsafe_option} is not allowed, use `allow_unsafe_options=True` to allow it.')