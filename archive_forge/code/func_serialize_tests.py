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
def serialize_tests(self) -> T.Tuple[str, str]:
    test_data = os.path.join(self.environment.get_scratch_dir(), 'meson_test_setup.dat')
    with open(test_data, 'wb') as datafile:
        self.write_test_file(datafile)
    benchmark_data = os.path.join(self.environment.get_scratch_dir(), 'meson_benchmark_setup.dat')
    with open(benchmark_data, 'wb') as datafile:
        self.write_benchmark_file(datafile)
    return (test_data, benchmark_data)