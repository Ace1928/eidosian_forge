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
def slots_to_dict(self: 'Git', exclude: Sequence[str]=()) -> Dict[str, Any]:
    return {s: getattr(self, s) for s in self.__slots__ if s not in exclude}