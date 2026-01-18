from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def check_direntry_issues(direntry_array: T.Union[T.Iterable[T.Union[str, bytes]], str, bytes]) -> None:
    import locale
    e = locale.getpreferredencoding()
    if e.upper() != 'UTF-8' and (not is_windows()):
        if isinstance(direntry_array, (str, bytes)):
            direntry_array = [direntry_array]
        for de in direntry_array:
            if is_ascii_string(de):
                continue
            mlog.warning(textwrap.dedent(f'\n                You are using {e!r} which is not a Unicode-compatible\n                locale but you are trying to access a file system entry called {de!r} which is\n                not pure ASCII. This may cause problems.\n                '))