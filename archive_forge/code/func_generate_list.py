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
def generate_list(func: T.Callable[..., T.Generator[_T, None, None]]) -> T.Callable[..., T.List[_T]]:

    @wraps(func)
    def wrapper(*args: T.Any, **kwargs: T.Any) -> T.List[_T]:
        return list(func(*args, **kwargs))
    return wrapper