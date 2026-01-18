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
def determine_windows_extra_paths(self, target: T.Union[build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, programs.ExternalProgram, mesonlib.File, str], extra_bdeps: T.Sequence[T.Union[build.BuildTarget, build.CustomTarget]]) -> T.List[str]:
    """On Windows there is no such thing as an rpath.

        We must determine all locations of DLLs that this exe
        links to and return them so they can be used in unit
        tests.
        """
    result: T.Set[str] = set()
    prospectives: T.Set[build.BuildTargetTypes] = set()
    if isinstance(target, build.BuildTarget):
        prospectives.update(target.get_transitive_link_deps())
        result.update(self.extract_dll_paths(target))
    for bdep in extra_bdeps:
        prospectives.add(bdep)
        if isinstance(bdep, build.BuildTarget):
            prospectives.update(bdep.get_transitive_link_deps())
    for ld in prospectives:
        dirseg = os.path.join(self.environment.get_build_dir(), self.get_target_dir(ld))
        result.add(dirseg)
    if isinstance(target, build.BuildTarget) and (not self.environment.machines.matches_build_machine(target.for_machine)):
        result.update(self.get_mingw_extra_paths(target))
    return list(result)