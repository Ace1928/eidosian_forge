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
def add_preserved_path_segment(self, infile: FileMaybeInTargetPrivateDir, outfiles: T.List[str], state: T.Union['Interpreter', 'ModuleState']) -> T.List[str]:
    result: T.List[str] = []
    in_abs = infile.absolute_path(state.environment.source_dir, state.environment.build_dir)
    assert os.path.isabs(self.preserve_path_from)
    rel = os.path.relpath(in_abs, self.preserve_path_from)
    path_segment = os.path.dirname(rel)
    for of in outfiles:
        result.append(os.path.join(path_segment, of))
    return result