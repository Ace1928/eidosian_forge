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
def get_transitive_link_deps(self) -> ImmutableListProtocol[BuildTargetTypes]:
    result: T.List[Target] = []
    for i in self.link_targets:
        result += i.get_all_link_deps()
    return result