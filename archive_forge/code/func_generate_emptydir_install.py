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
def generate_emptydir_install(self, d: InstallData) -> None:
    emptydir: T.List[build.EmptyDir] = self.build.get_emptydir()
    for e in emptydir:
        tag = e.install_tag or self.guess_install_tag(e.path)
        i = InstallEmptyDir(e.path, e.install_mode, e.subproject, tag)
        d.emptydir.append(i)