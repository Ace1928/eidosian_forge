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
def get_install_dir(self) -> T.Tuple[T.List[T.Union[str, Literal[False]]], T.List[T.Optional[str]], bool]:
    default_install_dir, default_install_dir_name = self.get_default_install_dir()
    outdirs: T.List[T.Union[str, Literal[False]]] = self.get_custom_install_dir()
    install_dir_names: T.List[T.Optional[str]]
    if outdirs and outdirs[0] != default_install_dir and (outdirs[0] is not True):
        custom_install_dir = True
        install_dir_names = [getattr(i, 'optname', None) for i in outdirs]
    else:
        custom_install_dir = False
        if outdirs:
            outdirs[0] = default_install_dir
        else:
            outdirs = [default_install_dir]
        install_dir_names = [default_install_dir_name] * len(outdirs)
    return (outdirs, install_dir_names, custom_install_dir)