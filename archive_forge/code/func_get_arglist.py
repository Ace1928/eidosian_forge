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
def get_arglist(self, inname: str) -> T.List[str]:
    plainname = os.path.basename(inname)
    basename = os.path.splitext(plainname)[0]
    return [x.replace('@BASENAME@', basename).replace('@PLAINNAME@', plainname) for x in self.arglist]