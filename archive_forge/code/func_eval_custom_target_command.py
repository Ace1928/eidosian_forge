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
def eval_custom_target_command(self, target: build.CustomTarget, absolute_outputs: bool=False) -> T.Tuple[T.List[str], T.List[str], T.List[str]]:
    source_root = self.build_to_src
    build_root = '.'
    outdir = self.get_custom_target_output_dir(target)
    if absolute_outputs:
        source_root = self.environment.get_source_dir()
        build_root = self.environment.get_build_dir()
        outdir = os.path.join(self.environment.get_build_dir(), outdir)
    outputs = [os.path.join(outdir, i) for i in target.get_outputs()]
    inputs = self.get_custom_target_sources(target)
    cmd: T.List[str] = []
    for i in target.command:
        if isinstance(i, build.BuildTarget):
            cmd += self.build_target_to_cmd_array(i)
            continue
        elif isinstance(i, build.CustomTarget):
            tmp = i.get_outputs()[0]
            i = os.path.join(self.get_custom_target_output_dir(i), tmp)
        elif isinstance(i, mesonlib.File):
            i = i.rel_to_builddir(self.build_to_src)
            if target.absolute_paths or absolute_outputs:
                i = os.path.join(self.environment.get_build_dir(), i)
        elif isinstance(i, str):
            if '@SOURCE_ROOT@' in i:
                i = i.replace('@SOURCE_ROOT@', source_root)
            if '@BUILD_ROOT@' in i:
                i = i.replace('@BUILD_ROOT@', build_root)
            if '@CURRENT_SOURCE_DIR@' in i:
                i = i.replace('@CURRENT_SOURCE_DIR@', os.path.join(source_root, target.subdir))
            if '@DEPFILE@' in i:
                if target.depfile is None:
                    msg = f'Custom target {target.name!r} has @DEPFILE@ but no depfile keyword argument.'
                    raise MesonException(msg)
                dfilename = os.path.join(outdir, target.depfile)
                i = i.replace('@DEPFILE@', dfilename)
            if '@PRIVATE_DIR@' in i:
                if target.absolute_paths:
                    pdir = self.get_target_private_dir_abs(target)
                else:
                    pdir = self.get_target_private_dir(target)
                i = i.replace('@PRIVATE_DIR@', pdir)
        else:
            raise RuntimeError(f'Argument {i} is of unknown type {type(i)}')
        cmd.append(i)
    values = mesonlib.get_filenames_templates_dict(inputs, outputs)
    cmd = mesonlib.substitute_values(cmd, values)
    cmd = [i.replace('\\', '/') for i in cmd]
    return (inputs, outputs, cmd)