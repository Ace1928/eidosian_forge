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
def process_vs_module_defs_kw(self, kwargs: T.Dict[str, T.Any]) -> None:
    if kwargs.get('vs_module_defs') is None:
        return
    path: T.Union[str, File, CustomTarget, CustomTargetIndex] = kwargs['vs_module_defs']
    if isinstance(path, str):
        if os.path.isabs(path):
            self.vs_module_defs = File.from_absolute_file(path)
        else:
            self.vs_module_defs = File.from_source_file(self.environment.source_dir, self.subdir, path)
    elif isinstance(path, File):
        self.vs_module_defs = path
    elif isinstance(path, (CustomTarget, CustomTargetIndex)):
        self.vs_module_defs = File.from_built_file(path.get_subdir(), path.get_filename())
    else:
        raise InvalidArguments('vs_module_defs must be either a string, a file object, a Custom Target, or a Custom Target Index')
    self.process_link_depends(path)