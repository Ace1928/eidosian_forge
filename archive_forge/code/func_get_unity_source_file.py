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
def get_unity_source_file(self, target: T.Union[build.BuildTarget, build.CustomTarget, build.CustomTargetIndex], suffix: str, number: int) -> mesonlib.File:
    osrc = f'{target.name}-unity{number}.{suffix}'
    return mesonlib.File.from_built_file(self.get_target_private_dir(target), osrc)