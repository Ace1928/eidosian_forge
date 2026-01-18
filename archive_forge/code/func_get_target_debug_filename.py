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
def get_target_debug_filename(self, target: build.BuildTarget) -> T.Optional[str]:
    assert isinstance(target, build.BuildTarget), target
    if target.get_debug_filename():
        debug_filename = target.get_debug_filename()
        return os.path.join(self.get_target_dir(target), debug_filename)
    else:
        return None