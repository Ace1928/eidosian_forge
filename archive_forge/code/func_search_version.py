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
def search_version(text: str) -> str:
    version_regex = re.compile('\n    (?<!                # Zero-width negative lookbehind assertion\n        (\n            \\d          # One digit\n            | \\.        # Or one period\n        )               # One occurrence\n    )\n    # Following pattern must not follow a digit or period\n    (\n        \\d{1,2}         # One or two digits\n        (\n            \\.\\d+       # Period and one or more digits\n        )+              # One or more occurrences\n        (\n            -[a-zA-Z0-9]+   # Hyphen and one or more alphanumeric\n        )?              # Zero or one occurrence\n    )                   # One occurrence\n    ', re.VERBOSE)
    match = version_regex.search(text)
    if match:
        return match.group(0)
    version_regex = re.compile('(\\d{1,4}\\.\\d{1,4}\\.?\\d{0,4})')
    match = version_regex.search(text)
    if match:
        return match.group(0)
    return 'unknown version'