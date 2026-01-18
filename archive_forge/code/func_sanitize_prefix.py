from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def sanitize_prefix(self, prefix: str) -> str:
    prefix = os.path.expanduser(prefix)
    if not os.path.isabs(prefix):
        raise MesonException(f'prefix value {prefix!r} must be an absolute path')
    if prefix.endswith('/') or prefix.endswith('\\'):
        if len(prefix) == 3 and prefix[1] == ':':
            pass
        elif len(prefix) == 1:
            pass
        else:
            prefix = prefix[:-1]
    return prefix