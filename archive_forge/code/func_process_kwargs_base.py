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
def process_kwargs_base(self, kwargs: T.Dict[str, T.Any]) -> None:
    if 'build_by_default' in kwargs:
        self.build_by_default = kwargs['build_by_default']
        if not isinstance(self.build_by_default, bool):
            raise InvalidArguments('build_by_default must be a boolean value.')
    if not self.build_by_default and kwargs.get('install', False):
        self.build_by_default = True
    self.set_option_overrides(self.parse_overrides(kwargs))