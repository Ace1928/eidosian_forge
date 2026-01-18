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
@lru_cache(maxsize=None)
def get_target_macos_dylib_install_name(ld) -> str:
    name = ['@rpath/', ld.prefix, ld.name]
    if ld.soversion is not None:
        name.append('.' + ld.soversion)
    name.append('.dylib')
    return ''.join(name)