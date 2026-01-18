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
def get_target_filename(self, t: T.Union[build.Target, build.CustomTargetIndex], *, warn_multi_output: bool=True) -> str:
    if isinstance(t, build.CustomTarget):
        if warn_multi_output and len(t.get_outputs()) != 1:
            mlog.warning(f'custom_target {t.name!r} has more than one output! Using the first one. Consider using `{t.name}[0]`.')
        filename = t.get_outputs()[0]
    elif isinstance(t, build.CustomTargetIndex):
        filename = t.get_outputs()[0]
    else:
        assert isinstance(t, build.BuildTarget), t
        filename = t.get_filename()
    return os.path.join(self.get_target_dir(t), filename)