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
def generate_subdir_install(self, d: InstallData) -> None:
    for sd in self.build.get_install_subdirs():
        if sd.from_source_dir:
            from_dir = self.environment.get_source_dir()
        else:
            from_dir = self.environment.get_build_dir()
        src_dir = os.path.join(from_dir, sd.source_subdir, sd.installable_subdir).rstrip('/')
        dst_dir = os.path.join(self.environment.get_prefix(), sd.install_dir)
        dst_name = os.path.join('{prefix}', sd.install_dir)
        if sd.install_dir != sd.install_dir_name:
            dst_name = sd.install_dir_name
        if not sd.strip_directory:
            dst_dir = os.path.join(dst_dir, os.path.basename(src_dir))
            dst_name = os.path.join(dst_name, os.path.basename(src_dir))
        tag = sd.install_tag or self.guess_install_tag(os.path.join(sd.install_dir, 'dummy'))
        i = SubdirInstallData(src_dir, dst_dir, dst_name, sd.install_mode, sd.exclude, sd.subproject, tag, follow_symlinks=sd.follow_symlinks)
        d.install_subdirs.append(i)