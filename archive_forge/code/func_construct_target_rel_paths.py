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
def construct_target_rel_paths(self, t: T.Union[build.Target, build.CustomTargetIndex], workdir: T.Optional[str]) -> T.List[str]:
    target_dir = self.get_target_dir(t)
    if isinstance(t, build.Executable) and workdir is None:
        target_dir = target_dir or '.'
    if isinstance(t, build.BuildTarget):
        outputs = [t.get_filename()]
    else:
        assert isinstance(t, (build.CustomTarget, build.CustomTargetIndex))
        outputs = t.get_outputs()
    outputs = [os.path.join(target_dir, x) for x in outputs]
    if workdir is not None:
        assert os.path.isabs(workdir)
        outputs = [os.path.join(self.environment.get_build_dir(), x) for x in outputs]
        outputs = [os.path.relpath(x, workdir) for x in outputs]
    return outputs