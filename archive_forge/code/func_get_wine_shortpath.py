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
def get_wine_shortpath(winecmd: T.List[str], wine_paths: T.List[str], workdir: T.Optional[str]=None) -> str:
    """
    WINEPATH size is limited to 1024 bytes which can easily be exceeded when
    adding the path to every dll inside build directory. See
    https://bugs.winehq.org/show_bug.cgi?id=45810.

    To shorten it as much as possible we use path relative to `workdir`
    where possible and convert absolute paths to Windows shortpath (e.g.
    "/usr/x86_64-w64-mingw32/lib" to "Z:\\usr\\X86_~FWL\\lib").

    This limitation reportedly has been fixed with wine >= 6.4
    """
    wine_paths = list(OrderedSet(wine_paths))
    wine_path = ';'.join(wine_paths)
    if len(wine_path) <= 1024:
        return wine_path
    from ..programs import ExternalProgram
    wine = ExternalProgram('wine', winecmd, silent=True)
    if version_compare(wine.get_version(), '>=6.4'):
        return wine_path
    rel_paths: T.List[str] = []
    if workdir:
        abs_paths: T.List[str] = []
        for p in wine_paths:
            try:
                rel = Path(p).relative_to(workdir)
                rel_paths.append(str(rel))
            except ValueError:
                abs_paths.append(p)
        wine_paths = abs_paths
    if wine_paths:
        with NamedTemporaryFile('w', suffix='.bat', encoding='utf-8', delete=False) as bat_file:
            bat_file.write('\n            @ECHO OFF\n            for %%x in (%*) do (\n                echo|set /p=;%~sx\n            )\n            ')
        try:
            stdout = subprocess.check_output(winecmd + ['cmd', '/C', bat_file.name] + wine_paths, encoding='utf-8', stderr=subprocess.DEVNULL)
            stdout = stdout.strip(';')
            if stdout:
                wine_paths = stdout.split(';')
            else:
                mlog.warning('Could not shorten WINEPATH: empty stdout')
        except subprocess.CalledProcessError as e:
            mlog.warning(f'Could not shorten WINEPATH: {str(e)}')
        finally:
            os.unlink(bat_file.name)
    wine_path = ';'.join(rel_paths + wine_paths)
    if len(wine_path) > 1024:
        mlog.warning('WINEPATH exceeds 1024 characters which could cause issues')
    return wine_path