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
def get_project_args(self, compiler: 'Compiler', project: str, for_machine: 'MachineChoice') -> T.List[str]:
    d = self.projects_args[for_machine]
    args = d.get(project)
    if not args:
        return []
    return args.get(compiler.get_language(), [])