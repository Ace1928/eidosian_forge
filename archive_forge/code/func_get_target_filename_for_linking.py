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
def get_target_filename_for_linking(self, target: T.Union[build.Target, build.CustomTargetIndex]) -> T.Optional[str]:
    if isinstance(target, build.SharedLibrary):
        link_lib = target.get_import_filename() or target.get_filename()
        if mesonlib.is_aix() and target.aix_so_archive:
            link_lib = re.sub('[.][a]([.]?([0-9]+))*([.]?([a-z]+))*', '.a', link_lib.replace('.so', '.a'))
        return Path(self.get_target_dir(target), link_lib).as_posix()
    elif isinstance(target, build.StaticLibrary):
        return Path(self.get_target_dir(target), target.get_filename()).as_posix()
    elif isinstance(target, (build.CustomTarget, build.CustomTargetIndex)):
        if not target.is_linkable_target():
            raise MesonException(f'Tried to link against custom target "{target.name}", which is not linkable.')
        return Path(self.get_target_dir(target), target.get_filename()).as_posix()
    elif isinstance(target, build.Executable):
        if target.import_filename:
            return Path(self.get_target_dir(target), target.get_import_filename()).as_posix()
        else:
            return None
    raise AssertionError(f'BUG: Tried to link to {target!r} which is not linkable')