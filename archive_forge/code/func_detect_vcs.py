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
def detect_vcs(source_dir: T.Union[str, Path]) -> T.Optional[T.Dict[str, str]]:
    vcs_systems = [{'name': 'git', 'cmd': 'git', 'repo_dir': '.git', 'get_rev': 'git describe --dirty=+ --always', 'rev_regex': '(.*)', 'dep': '.git/logs/HEAD'}, {'name': 'mercurial', 'cmd': 'hg', 'repo_dir': '.hg', 'get_rev': 'hg id -i', 'rev_regex': '(.*)', 'dep': '.hg/dirstate'}, {'name': 'subversion', 'cmd': 'svn', 'repo_dir': '.svn', 'get_rev': 'svn info', 'rev_regex': 'Revision: (.*)', 'dep': '.svn/wc.db'}, {'name': 'bazaar', 'cmd': 'bzr', 'repo_dir': '.bzr', 'get_rev': 'bzr revno', 'rev_regex': '(.*)', 'dep': '.bzr'}]
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)
    parent_paths_and_self = collections.deque(source_dir.parents)
    parent_paths_and_self.appendleft(source_dir)
    for curdir in parent_paths_and_self:
        for vcs in vcs_systems:
            if Path.is_dir(curdir.joinpath(vcs['repo_dir'])) and shutil.which(vcs['cmd']):
                vcs['wc_dir'] = str(curdir)
                return vcs
    return None