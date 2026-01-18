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
def polish_url(cls, url: str, is_cygwin: Union[None, bool]=None) -> PathLike:
    """Remove any backslashes from urls to be written in config files.

        Windows might create config files containing paths with backslashes,
        but git stops liking them as it will escape the backslashes. Hence we
        undo the escaping just to be sure.
        """
    if is_cygwin is None:
        is_cygwin = cls.is_cygwin()
    if is_cygwin:
        url = cygpath(url)
    else:
        url = os.path.expandvars(url)
        if url.startswith('~'):
            url = os.path.expanduser(url)
        url = url.replace('\\\\', '\\').replace('\\', '/')
    return url