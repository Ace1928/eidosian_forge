from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def flatten_command(self, cmd: T.Sequence[T.Union[str, File, programs.ExternalProgram, BuildTargetTypes]]) -> T.List[T.Union[str, File, BuildTarget, 'CustomTarget']]:
    cmd = listify(cmd)
    final_cmd: T.List[T.Union[str, File, BuildTarget, 'CustomTarget']] = []
    for c in cmd:
        if isinstance(c, str):
            final_cmd.append(c)
        elif isinstance(c, File):
            self.depend_files.append(c)
            final_cmd.append(c)
        elif isinstance(c, programs.ExternalProgram):
            if not c.found():
                raise InvalidArguments('Tried to use not-found external program in "command"')
            path = c.get_path()
            if os.path.isabs(path):
                self.depend_files.append(File.from_absolute_file(path))
            final_cmd += c.get_command()
        elif isinstance(c, (BuildTarget, CustomTarget)):
            self.dependencies.append(c)
            final_cmd.append(c)
        elif isinstance(c, CustomTargetIndex):
            FeatureNew.single_use('CustomTargetIndex for command argument', '0.60', self.subproject)
            self.dependencies.append(c.target)
            final_cmd += self.flatten_command(File.from_built_file(c.get_subdir(), c.get_filename()))
        elif isinstance(c, list):
            final_cmd += self.flatten_command(c)
        else:
            raise InvalidArguments(f'Argument {c!r} in "command" is invalid')
    return final_cmd