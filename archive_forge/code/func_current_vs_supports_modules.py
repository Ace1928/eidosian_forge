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
def current_vs_supports_modules() -> bool:
    vsver = os.environ.get('VSCMD_VER', '')
    nums = vsver.split('.', 2)
    major = int(nums[0])
    if major >= 17:
        return True
    if major == 16 and int(nums[1]) >= 10:
        return True
    return vsver.startswith('16.9.0') and '-pre.' in vsver