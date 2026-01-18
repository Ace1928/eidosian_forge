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
def determine_ext_objs(self, objects: build.ExtractedObjects, proj_dir_to_build_root: str='') -> T.List[str]:
    obj_list, _ = self._flatten_object_list(objects.target, [objects], proj_dir_to_build_root)
    return list(dict.fromkeys(obj_list))