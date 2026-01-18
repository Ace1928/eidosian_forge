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
def get_dep_outname(self, infilenames):
    if self.depfile is None:
        raise InvalidArguments('Tried to get depfile name for custom_target that does not have depfile defined.')
    if infilenames:
        plainname = os.path.basename(infilenames[0])
        basename = os.path.splitext(plainname)[0]
        return self.depfile.replace('@BASENAME@', basename).replace('@PLAINNAME@', plainname)
    else:
        if '@BASENAME@' in self.depfile or '@PLAINNAME@' in self.depfile:
            raise InvalidArguments('Substitution in depfile for custom_target that does not have an input file.')
        return self.depfile