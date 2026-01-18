from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def replace_extra_args(self, args: T.List[str], genlist: 'build.GeneratedList') -> T.List[str]:
    final_args: T.List[str] = []
    for a in args:
        if a == '@EXTRA_ARGS@':
            final_args += genlist.get_extra_args()
        else:
            final_args.append(a)
    return final_args