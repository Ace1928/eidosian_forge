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
def generate_regen_info(self) -> None:
    deps = self.get_regen_filelist()
    regeninfo = RegenInfo(self.environment.get_source_dir(), self.environment.get_build_dir(), deps)
    filename = os.path.join(self.environment.get_scratch_dir(), 'regeninfo.dump')
    with open(filename, 'wb') as f:
        pickle.dump(regeninfo, f)