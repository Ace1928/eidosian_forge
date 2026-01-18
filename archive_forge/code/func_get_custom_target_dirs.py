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
def get_custom_target_dirs(self, target: build.CustomTarget, compiler: 'Compiler', *, absolute_path: bool=False) -> T.List[str]:
    custom_target_include_dirs: T.List[str] = []
    for i in target.get_generated_sources():
        if not isinstance(i, (build.CustomTarget, build.CustomTargetIndex)):
            continue
        idir = self.get_normpath_target(self.get_custom_target_output_dir(i))
        if not idir:
            idir = '.'
        if absolute_path:
            idir = os.path.join(self.environment.get_build_dir(), idir)
        if idir not in custom_target_include_dirs:
            custom_target_include_dirs.append(idir)
    return custom_target_include_dirs