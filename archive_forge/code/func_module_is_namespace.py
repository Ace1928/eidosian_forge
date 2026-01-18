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
def module_is_namespace(mod: ModuleType) -> bool:
    """Is the module object `mod` a PEP420 namespace module?"""
    return hasattr(mod, '__path__') and getattr(mod, '__file__', None) is None