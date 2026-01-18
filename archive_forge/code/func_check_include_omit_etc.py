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
def check_include_omit_etc(self, filename: str, frame: FrameType | None) -> str | None:
    """Check a file name against the include, omit, etc, rules.

        Returns a string or None.  String means, don't trace, and is the reason
        why.  None means no reason found to not trace.

        """
    modulename = name_for_module(filename, frame)
    if self.source_match or self.source_pkgs_match:
        extra = ''
        ok = False
        if self.source_pkgs_match:
            if self.source_pkgs_match.match(modulename):
                ok = True
                if modulename in self.source_pkgs_unmatched:
                    self.source_pkgs_unmatched.remove(modulename)
            else:
                extra = f'module {modulename!r} '
        if not ok and self.source_match:
            if self.source_match.match(filename):
                ok = True
        if not ok:
            return extra + 'falls outside the --source spec'
        if self.third_match.match(filename) and (not self.source_in_third_match.match(filename)):
            return 'inside --source, but is third-party'
    elif self.include_match:
        if not self.include_match.match(filename):
            return 'falls outside the --include trees'
    else:
        if self.cover_match.match(filename):
            return 'is part of coverage.py'
        if self.pylib_match and self.pylib_match.match(filename):
            return 'is in the stdlib'
        if self.third_match.match(filename):
            return 'is a third-party module'
    if self.omit_match and self.omit_match.match(filename):
        return 'is inside an --omit pattern'
    try:
        filename.encode('utf-8')
    except UnicodeEncodeError:
        return 'non-encodable filename'
    return None