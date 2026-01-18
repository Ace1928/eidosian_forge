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
def get_no_stdlib_args(self, target: 'build.BuildTarget', compiler: 'Compiler') -> T.List[str]:
    if compiler.language in self.build.stdlibs[target.for_machine]:
        return compiler.get_no_stdinc_args()
    return []