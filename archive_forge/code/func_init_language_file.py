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
def init_language_file(suffix: str, unity_file_number: int) -> T.TextIO:
    unity_src = self.get_unity_source_file(target, suffix, unity_file_number)
    outfileabs = unity_src.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
    outfileabs_tmp = outfileabs + '.tmp'
    abs_files.append(outfileabs)
    outfileabs_tmp_dir = os.path.dirname(outfileabs_tmp)
    if not os.path.exists(outfileabs_tmp_dir):
        os.makedirs(outfileabs_tmp_dir)
    result.append(unity_src)
    return open(outfileabs_tmp, 'w', encoding='utf-8')