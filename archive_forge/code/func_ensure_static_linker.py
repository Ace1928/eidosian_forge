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
def ensure_static_linker(self, compiler: Compiler) -> None:
    if self.static_linker[compiler.for_machine] is None and compiler.needs_static_linker():
        self.static_linker[compiler.for_machine] = detect_static_linker(self.environment, compiler)