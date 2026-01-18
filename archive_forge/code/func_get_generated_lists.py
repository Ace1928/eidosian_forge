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
def get_generated_lists(self) -> T.List[GeneratedList]:
    genlists: T.List[GeneratedList] = []
    for c in self.sources:
        if isinstance(c, GeneratedList):
            genlists.append(c)
    return genlists