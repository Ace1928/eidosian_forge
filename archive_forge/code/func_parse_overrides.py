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
def parse_overrides(kwargs: T.Dict[str, T.Any]) -> T.Dict[OptionKey, str]:
    opts = kwargs.get('override_options', [])
    if isinstance(opts, dict):
        return T.cast('T.Dict[OptionKey, str]', opts)
    result: T.Dict[OptionKey, str] = {}
    overrides = stringlistify(opts)
    for o in overrides:
        if '=' not in o:
            raise InvalidArguments('Overrides must be of form "key=value"')
        k, v = o.split('=', 1)
        key = OptionKey.from_string(k.strip())
        v = v.strip()
        result[key] = v
    return result