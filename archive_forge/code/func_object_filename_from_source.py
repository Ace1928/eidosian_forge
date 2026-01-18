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
def object_filename_from_source(self, target: build.BuildTarget, source: 'FileOrString', targetdir: T.Optional[str]=None) -> str:
    assert isinstance(source, mesonlib.File)
    if isinstance(target, build.CompileTarget):
        return target.sources_map[source]
    build_dir = self.environment.get_build_dir()
    rel_src = source.rel_to_builddir(self.build_to_src)
    if rel_src.endswith(('.vala', '.gs')):
        if source.is_built:
            if os.path.isabs(rel_src):
                rel_src = rel_src[len(build_dir) + 1:]
            rel_src = os.path.relpath(rel_src, self.get_target_private_dir(target))
        else:
            rel_src = os.path.basename(rel_src)
        gen_source = 'meson-generated_' + rel_src[:-5] + '.c'
    elif source.is_built:
        if os.path.isabs(rel_src):
            rel_src = rel_src[len(build_dir) + 1:]
        gen_source = 'meson-generated_' + os.path.relpath(rel_src, self.get_target_private_dir(target))
    elif os.path.isabs(rel_src):
        gen_source = rel_src
    else:
        gen_source = os.path.relpath(os.path.join(build_dir, rel_src), os.path.join(self.environment.get_source_dir(), target.get_subdir()))
    machine = self.environment.machines[target.for_machine]
    ret = self.canonicalize_filename(gen_source) + '.' + machine.get_object_suffix()
    if targetdir is not None:
        return os.path.join(targetdir, ret)
    return ret