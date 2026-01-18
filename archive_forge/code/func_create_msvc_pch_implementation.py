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
def create_msvc_pch_implementation(self, target: build.BuildTarget, lang: str, pch_header: str) -> str:
    impl_name = f'meson_pch-{lang}.{lang}'
    pch_rel_to_build = os.path.join(self.get_target_private_dir(target), impl_name)
    pch_file = os.path.join(self.build_dir, pch_rel_to_build)
    os.makedirs(os.path.dirname(pch_file), exist_ok=True)
    content = f'#include "{os.path.basename(pch_header)}"'
    pch_file_tmp = pch_file + '.tmp'
    with open(pch_file_tmp, 'w', encoding='utf-8') as f:
        f.write(content)
    mesonlib.replace_if_different(pch_file, pch_file_tmp)
    return pch_rel_to_build