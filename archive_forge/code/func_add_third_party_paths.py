from __future__ import annotations
import importlib.util
import inspect
import itertools
import os
import platform
import re
import sys
import sysconfig
import traceback
from types import FrameType, ModuleType
from typing import (
from coverage import env
from coverage.disposition import FileDisposition, disposition_init
from coverage.exceptions import CoverageException, PluginError
from coverage.files import TreeMatcher, GlobMatcher, ModuleMatcher
from coverage.files import prep_patterns, find_python_files, canonical_filename
from coverage.misc import sys_modules_saved
from coverage.python import source_for_file, source_for_morf
from coverage.types import TFileDisposition, TMorf, TWarnFn, TDebugCtl
def add_third_party_paths(paths: set[str]) -> None:
    """Add locations for third-party packages to the set `paths`."""
    scheme_names = set(sysconfig.get_scheme_names())
    for scheme in scheme_names:
        better_scheme = 'pypy_posix' if scheme == 'pypy' else scheme
        if os.name in better_scheme.split('_'):
            config_paths = sysconfig.get_paths(scheme)
            for path_name in ['platlib', 'purelib', 'scripts']:
                paths.add(config_paths[path_name])