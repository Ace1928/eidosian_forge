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
def construct_id_from_path(subdir: str, name: str, type_suffix: str) -> str:
    """Construct target ID from subdir, name and type suffix.

        This helper function is made public mostly for tests."""
    name_part = name.replace('/', '@').replace('\\', '@')
    assert not has_path_sep(type_suffix)
    my_id = name_part + type_suffix
    if subdir:
        subdir_part = Target._get_id_hash(subdir)
        return subdir_part + '@@' + my_id
    return my_id