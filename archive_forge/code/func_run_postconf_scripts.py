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
def run_postconf_scripts(self) -> None:
    from ..scripts.meson_exe import run_exe
    env = {'MESON_SOURCE_ROOT': self.environment.get_source_dir(), 'MESON_BUILD_ROOT': self.environment.get_build_dir(), 'MESONINTROSPECT': self.get_introspect_command()}
    for s in self.build.postconf_scripts:
        name = ' '.join(s.cmd_args)
        mlog.log(f'Running postconf script {name!r}')
        rc = run_exe(s, env)
        if rc != 0:
            raise MesonException(f"Postconf script '{name}' failed with exit code {rc}.")