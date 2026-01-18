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
def generate_header_install(self, d: InstallData) -> None:
    incroot = self.environment.get_includedir()
    headers = self.build.get_headers()
    srcdir = self.environment.get_source_dir()
    builddir = self.environment.get_build_dir()
    for h in headers:
        outdir = outdir_name = h.get_custom_install_dir()
        if outdir is None:
            subdir = h.get_install_subdir()
            if subdir is None:
                outdir = incroot
                outdir_name = '{includedir}'
            else:
                outdir = os.path.join(incroot, subdir)
                outdir_name = os.path.join('{includedir}', subdir)
        for f in h.get_sources():
            abspath = f.absolute_path(srcdir, builddir)
            i = InstallDataBase(abspath, outdir, outdir_name, h.get_custom_install_mode(), h.subproject, tag='devel', follow_symlinks=h.follow_symlinks)
            d.headers.append(i)