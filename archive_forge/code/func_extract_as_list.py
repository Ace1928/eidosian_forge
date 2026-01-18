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
def extract_as_list(dict_object: T.Dict[_T, _U], key: _T, pop: bool=False) -> T.List[_U]:
    """
    Extracts all values from given dict_object and listifies them.
    """
    fetch: T.Callable[[_T], _U] = dict_object.get
    if pop:
        fetch = dict_object.pop
    return listify(fetch(key) or [], flatten=True)