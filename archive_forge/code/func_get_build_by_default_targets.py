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
def get_build_by_default_targets(self) -> 'T.OrderedDict[str, T.Union[build.BuildTarget, build.CustomTarget]]':
    result: 'T.OrderedDict[str, T.Union[build.BuildTarget, build.CustomTarget]]' = OrderedDict()
    for name, b in self.build.get_targets().items():
        if b.build_by_default:
            result[name] = b
    return result