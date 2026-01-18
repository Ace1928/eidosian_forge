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
def build_target_to_cmd_array(self, bt: T.Union[build.BuildTarget, programs.ExternalProgram]) -> T.List[str]:
    if isinstance(bt, build.BuildTarget):
        arr = [os.path.join(self.environment.get_build_dir(), self.get_target_filename(bt))]
    else:
        arr = bt.get_command()
    return arr