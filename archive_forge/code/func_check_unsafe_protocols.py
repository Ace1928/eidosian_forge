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
def check_unsafe_protocols(cls, url: str) -> None:
    """Check for unsafe protocols.

        Apart from the usual protocols (http, git, ssh),
        Git allows "remote helpers" that have the form ``<transport>::<address>``.
        One of these helpers (``ext::``) can be used to invoke any arbitrary command.

        See:

        - https://git-scm.com/docs/gitremote-helpers
        - https://git-scm.com/docs/git-remote-ext
        """
    match = cls.re_unsafe_protocol.match(url)
    if match:
        protocol = match.group(1)
        raise UnsafeProtocolError(f'The `{protocol}::` protocol looks suspicious, use `allow_unsafe_protocols=True` to allow it.')