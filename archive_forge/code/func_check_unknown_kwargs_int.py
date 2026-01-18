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
def check_unknown_kwargs_int(self, kwargs, known_kwargs):
    unknowns = []
    for k in kwargs:
        if k == 'language_args':
            continue
        if k not in known_kwargs:
            unknowns.append(k)
    if len(unknowns) > 0:
        mlog.warning('Unknown keyword argument(s) in target {}: {}.'.format(self.name, ', '.join(unknowns)))