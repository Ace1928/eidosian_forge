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
def get_object_header(self, ref: str) -> Tuple[str, str, int]:
    """Use this method to quickly examine the type and size of the object behind
        the given ref.

        :note: The method will only suffer from the costs of command invocation
            once and reuses the command in subsequent calls.

        :return: (hexsha, type_string, size_as_int)
        """
    cmd = self._get_persistent_cmd('cat_file_header', 'cat_file', batch_check=True)
    return self.__get_object_header(cmd, ref)