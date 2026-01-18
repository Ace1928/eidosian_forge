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
def check_module_linking(self):
    """
        Warn if shared modules are linked with target: (link_with) #2865
        """
    for link_target in self.link_targets:
        if isinstance(link_target, SharedModule) and (not link_target.force_soname):
            if self.environment.machines[self.for_machine].is_darwin():
                raise MesonException(f'target {self.name} links against shared module {link_target.name}. This is not permitted on OSX')
            elif self.environment.machines[self.for_machine].is_android() and isinstance(self, SharedModule):
                link_target.force_soname = True
            else:
                mlog.deprecation(f"target {self.name} links against shared module {link_target.name}, which is incorrect.\n             This will be an error in the future, so please use shared_library() for {link_target.name} instead.\n             If shared_module() was used for {link_target.name} because it has references to undefined symbols,\n             use shared_library() with `override_options: ['b_lundef=false']` instead.")
                link_target.force_soname = True