from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def lookup_binary_entry(self, for_machine: MachineChoice, name: str) -> T.Optional[T.List[str]]:
    return self.binaries[for_machine].lookup_entry(name)