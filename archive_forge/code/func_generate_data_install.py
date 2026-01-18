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
def generate_data_install(self, d: InstallData) -> None:
    data = self.build.get_data()
    srcdir = self.environment.get_source_dir()
    builddir = self.environment.get_build_dir()
    for de in data:
        assert isinstance(de, build.Data)
        subdir = de.install_dir
        subdir_name = de.install_dir_name
        for src_file, dst_name in zip(de.sources, de.rename):
            assert isinstance(src_file, mesonlib.File)
            dst_abs = os.path.join(subdir, dst_name)
            dstdir_name = os.path.join(subdir_name, dst_name)
            tag = de.install_tag or self.guess_install_tag(dst_abs)
            i = InstallDataBase(src_file.absolute_path(srcdir, builddir), dst_abs, dstdir_name, de.install_mode, de.subproject, tag=tag, data_type=de.data_type, follow_symlinks=de.follow_symlinks)
            d.data.append(i)