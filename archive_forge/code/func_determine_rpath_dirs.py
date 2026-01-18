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
def determine_rpath_dirs(self, target: T.Union[build.BuildTarget, build.CustomTarget, build.CustomTargetIndex]) -> T.Tuple[str, ...]:
    result: OrderedSet[str]
    if self.environment.coredata.get_option(OptionKey('layout')) == 'mirror':
        result = OrderedSet(target.get_link_dep_subdirs())
    else:
        result = OrderedSet()
        result.add('meson-out')
    if isinstance(target, build.BuildTarget):
        result.update(self.rpaths_for_non_system_absolute_shared_libraries(target))
        target.rpath_dirs_to_remove.update([d.encode('utf-8') for d in result])
    return tuple(result)