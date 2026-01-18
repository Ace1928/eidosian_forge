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
def get_prelinker(self):
    if self.link_language:
        comp = self.all_compilers[self.link_language]
        return comp
    for l in clink_langs:
        if l in self.compilers:
            try:
                prelinker = self.all_compilers[l]
            except KeyError:
                raise MesonException(f'Could not get a prelinker linker for build target {self.name!r}. Requires a compiler for language "{l}", but that is not a project language.')
            return prelinker
    raise MesonException(f'Could not determine prelinker for {self.name!r}.')