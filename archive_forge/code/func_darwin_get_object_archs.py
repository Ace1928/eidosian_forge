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
@lru_cache(maxsize=None)
def darwin_get_object_archs(objpath: str) -> 'ImmutableListProtocol[str]':
    """
    For a specific object (executable, static library, dylib, etc), run `lipo`
    to fetch the list of archs supported by it. Supports both thin objects and
    'fat' objects.
    """
    _, stdo, stderr = Popen_safe(['lipo', '-info', objpath])
    if not stdo:
        mlog.debug(f'lipo {objpath}: {stderr}')
        return None
    stdo = stdo.rsplit(': ', 1)[1]
    map_arch = {'i386': 'x86', 'arm64': 'aarch64', 'arm64e': 'aarch64', 'ppc7400': 'ppc', 'ppc970': 'ppc'}
    lipo_archs = stdo.split()
    meson_archs = [map_arch.get(lipo_arch, lipo_arch) for lipo_arch in lipo_archs]
    if 'armv7' in stdo:
        meson_archs.append('arm')
    return meson_archs