from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
@classmethod
@lru_cache(maxsize=None)
def extract_dll_paths(cls, target: build.BuildTarget) -> T.Set[str]:
    """Find paths to all DLLs needed for a given target, since
        we link against import libs, and we don't know the actual
        path of the DLLs.

        1. If there are DLLs in the same directory than the .lib dir, use it
        2. If there is a sibbling directory named 'bin' with DLLs in it, use it
        """
    results = set()
    for dep in target.external_deps:
        if dep.type_name == 'pkgconfig':
            bindir = dep.get_variable(pkgconfig='bindir', default_value='')
            if bindir:
                results.add(bindir)
                continue
        results.update(filter(None, map(cls.search_dll_path, dep.link_args)))
    for i in chain(target.link_targets, target.link_whole_targets):
        if isinstance(i, build.BuildTarget):
            results.update(cls.extract_dll_paths(i))
    return results