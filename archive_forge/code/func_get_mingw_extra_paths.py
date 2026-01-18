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
def get_mingw_extra_paths(self, target: build.BuildTarget) -> T.List[str]:
    paths: OrderedSet[str] = OrderedSet()
    root = self.environment.properties[target.for_machine].get_root()
    if root:
        paths.add(os.path.join(root, 'bin'))
    sys_root = self.environment.properties[target.for_machine].get_sys_root()
    if sys_root:
        paths.add(os.path.join(sys_root, 'bin'))
    if isinstance(target, build.BuildTarget):
        for cc in target.compilers.values():
            paths.update(cc.get_program_dirs(self.environment))
            paths.update(cc.get_library_dirs(self.environment))
    return list(paths)