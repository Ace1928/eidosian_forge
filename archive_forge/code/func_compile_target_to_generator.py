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
def compile_target_to_generator(self, target: build.CompileTarget) -> build.GeneratedList:
    all_sources = T.cast('_ALL_SOURCES_TYPE', target.sources) + T.cast('_ALL_SOURCES_TYPE', target.generated)
    return self.compiler_to_generator(target, target.compiler, all_sources, target.output_templ, target.depends)