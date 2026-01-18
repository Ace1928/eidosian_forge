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
@staticmethod
def is_parent_path(parent: str, trial: str) -> bool:
    try:
        common = os.path.commonpath((parent, trial))
    except ValueError:
        return False
    return pathlib.PurePath(common) == pathlib.PurePath(parent)