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
def dict_to_slots_and__excluded_are_none(self: object, d: Mapping[str, Any], excluded: Sequence[str]=()) -> None:
    for k, v in d.items():
        setattr(self, k, v)
    for k in excluded:
        setattr(self, k, None)