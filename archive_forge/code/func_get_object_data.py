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
def get_object_data(self, ref: str) -> Tuple[str, str, int, bytes]:
    """As get_object_header, but returns object data as well.

        :return: (hexsha, type_string, size_as_int, data_string)
        :note: Not threadsafe.
        """
    hexsha, typename, size, stream = self.stream_object_data(ref)
    data = stream.read(size)
    del stream
    return (hexsha, typename, size, data)