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
def check_can_link_together(self, t: BuildTargetTypes) -> None:
    links_with_rust_abi = isinstance(t, BuildTarget) and t.uses_rust_abi()
    if not self.uses_rust() and links_with_rust_abi:
        raise InvalidArguments(f'Try to link Rust ABI library {t.name!r} with a non-Rust target {self.name!r}')
    if self.for_machine is not t.for_machine and (not links_with_rust_abi or t.rust_crate_type != 'proc-macro'):
        msg = f'Tried to tied to mix a {t.for_machine} library ("{t.name}") with a {self.for_machine} target "{self.name}"'
        if self.environment.is_cross_build():
            raise InvalidArguments(msg + ' This is not possible in a cross build.')
        else:
            mlog.warning(msg + ' This will fail in cross build.')