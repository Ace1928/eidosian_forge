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
def get_global_link_args(self, compiler: 'Compiler', for_machine: 'MachineChoice') -> T.List[str]:
    d = self.global_link_args[for_machine]
    return d.get(compiler.get_language(), [])