from __future__ import annotations
from pathlib import Path
from collections import deque
from contextlib import suppress
from copy import deepcopy
from fnmatch import fnmatch
import argparse
import asyncio
import datetime
import enum
import json
import multiprocessing
import os
import pickle
import platform
import random
import re
import signal
import subprocess
import shlex
import sys
import textwrap
import time
import typing as T
import unicodedata
import xml.etree.ElementTree as et
from . import build
from . import environment
from . import mlog
from .coredata import MesonVersionMismatchException, major_versions_differ
from .coredata import version as coredata_version
from .mesonlib import (MesonException, OptionKey, OrderedSet, RealPathAction,
from .mintro import get_infodir, load_info_file
from .programs import ExternalProgram
from .backend.backends import TestProtocol, TestSerialisation
def rebuild_deps(ninja: T.List[str], wd: str, tests: T.List[TestSerialisation]) -> bool:

    def convert_path_to_target(path: str) -> str:
        path = os.path.relpath(path, wd)
        if os.sep != '/':
            path = path.replace(os.sep, '/')
        return path
    assert len(ninja) > 0
    depends: T.Set[str] = set()
    targets: T.Set[str] = set()
    intro_targets: T.Dict[str, T.List[str]] = {}
    for target in load_info_file(get_infodir(wd), kind='targets'):
        intro_targets[target['id']] = [convert_path_to_target(f) for f in target['filename']]
    for t in tests:
        for d in t.depends:
            if d in depends:
                continue
            depends.update(d)
            targets.update(intro_targets[d])
    ret = subprocess.run(ninja + ['-C', wd] + sorted(targets)).returncode
    if ret != 0:
        print(f'Could not rebuild {wd}')
        return False
    return True