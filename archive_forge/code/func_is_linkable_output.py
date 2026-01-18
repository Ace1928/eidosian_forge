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
def is_linkable_output(self, output: str) -> bool:
    if output.endswith(('.a', '.dll', '.lib', '.so', '.dylib')):
        return True
    if re.search('\\.so(\\.\\d+)*$', output):
        return True
    return False