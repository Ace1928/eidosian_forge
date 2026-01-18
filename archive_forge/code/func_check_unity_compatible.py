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
def check_unity_compatible(self) -> None:
    cmpsrcs = self.classify_all_sources(self.target.sources, self.target.generated)
    extracted_cmpsrcs = self.classify_all_sources(self.srclist, self.genlist)
    for comp, srcs in extracted_cmpsrcs.items():
        if set(srcs) != set(cmpsrcs[comp]):
            raise MesonException('Single object files cannot be extracted in Unity builds. You can only extract all the object files for each compiler at once.')