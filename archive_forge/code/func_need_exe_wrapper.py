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
def need_exe_wrapper(self, for_machine: MachineChoice=MachineChoice.HOST):
    value = self.properties[for_machine].get('needs_exe_wrapper', None)
    if value is not None:
        return value
    return not machine_info_can_run(self.machines[for_machine])