from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def do_conf_str(src: str, data: T.List[str], confdata: 'ConfigurationData', variable_format: Literal['meson', 'cmake', 'cmake@'], subproject: T.Optional[SubProject]=None) -> T.Tuple[T.List[str], T.Set[str], bool]:

    def line_is_valid(line: str, variable_format: str) -> bool:
        if variable_format == 'meson':
            if '#cmakedefine' in line:
                return False
        elif '#mesondefine' in line:
            return False
        return True
    regex = get_variable_regex(variable_format)
    search_token = '#mesondefine'
    if variable_format != 'meson':
        search_token = '#cmakedefine'
    result: T.List[str] = []
    missing_variables: T.Set[str] = set()
    confdata_useless = not confdata.keys()
    for line in data:
        if line.lstrip().startswith(search_token):
            confdata_useless = False
            line = do_define(regex, line, confdata, variable_format, subproject)
        else:
            if not line_is_valid(line, variable_format):
                raise MesonException(f'Format error in {src}: saw "{line.strip()}" when format set to "{variable_format}"')
            line, missing = do_replacement(regex, line, variable_format, confdata)
            missing_variables.update(missing)
            if missing:
                confdata_useless = False
        result.append(line)
    return (result, missing_variables, confdata_useless)