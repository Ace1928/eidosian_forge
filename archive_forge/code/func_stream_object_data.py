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
def stream_object_data(self, ref: str) -> Tuple[str, str, int, 'Git.CatFileContentStream']:
    """As get_object_header, but returns the data as a stream.

        :return: (hexsha, type_string, size_as_int, stream)
        :note: This method is not threadsafe, you need one independent Command instance per thread to be safe!
        """
    cmd = self._get_persistent_cmd('cat_file_all', 'cat_file', batch=True)
    hexsha, typename, size = self.__get_object_header(cmd, ref)
    cmd_stdout = cmd.stdout if cmd.stdout is not None else io.BytesIO()
    return (hexsha, typename, size, self.CatFileContentStream(size, cmd_stdout))