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
def get_custom_target_provided_by_generated_source(self, generated_source: build.CustomTarget) -> 'ImmutableListProtocol[str]':
    libs: T.List[str] = []
    for f in generated_source.get_outputs():
        if self.environment.is_library(f):
            libs.append(os.path.join(self.get_target_dir(generated_source), f))
    return libs