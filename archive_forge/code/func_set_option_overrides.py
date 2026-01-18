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
def set_option_overrides(self, option_overrides: T.Dict[OptionKey, str]) -> None:
    self.options.overrides = {}
    for k, v in option_overrides.items():
        if k.lang:
            self.options.overrides[k.evolve(machine=self.for_machine)] = v
        else:
            self.options.overrides[k] = v