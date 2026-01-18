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
def get_target_dependencies(self) -> T.List[T.Union[SourceOutputs, str]]:
    deps: T.List[T.Union[SourceOutputs, str]] = []
    deps.extend(self.dependencies)
    deps.extend(self.extra_depends)
    for c in self.sources:
        if isinstance(c, CustomTargetIndex):
            deps.append(c.target)
        elif not isinstance(c, programs.ExternalProgram):
            deps.append(c)
    return deps