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
def get_langs_used_by_deps(self) -> T.List[str]:
    """
        Sometimes you want to link to a C++ library that exports C API, which
        means the linker must link in the C++ stdlib, and we must use a C++
        compiler for linking. The same is also applicable for objc/objc++, etc,
        so we can keep using clink_langs for the priority order.

        See: https://github.com/mesonbuild/meson/issues/1653
        """
    langs: T.List[str] = []
    for dep in self.external_deps:
        if dep.language is None:
            continue
        if dep.language not in langs:
            langs.append(dep.language)
    for link_target in itertools.chain(self.link_targets, self.link_whole_targets):
        if isinstance(link_target, (CustomTarget, CustomTargetIndex)):
            continue
        for language in link_target.compilers:
            if language not in langs:
                langs.append(language)
    return langs