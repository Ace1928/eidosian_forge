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
@lru_cache(maxsize=None)
def get_target_generated_dir(self, target: T.Union[build.BuildTarget, build.CustomTarget, build.CustomTargetIndex], gensrc: T.Union[build.CustomTarget, build.CustomTargetIndex, build.GeneratedList], src: str) -> str:
    """
        Takes a BuildTarget, a generator source (CustomTarget or GeneratedList),
        and a generated source filename.
        Returns the full path of the generated source relative to the build root
        """
    if isinstance(gensrc, (build.CustomTarget, build.CustomTargetIndex)):
        return os.path.join(self.get_target_dir(gensrc), src)
    return os.path.join(self.get_target_private_dir(target), src)