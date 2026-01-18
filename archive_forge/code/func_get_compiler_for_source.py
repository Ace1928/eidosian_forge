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
def get_compiler_for_source(compilers: T.Iterable['Compiler'], src: 'FileOrString') -> 'Compiler':
    """Given a set of compilers and a source, find the compiler for that source type."""
    for comp in compilers:
        if comp.can_compile(src):
            return comp
    raise MesonException(f'No specified compiler can handle file {src!s}')