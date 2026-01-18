from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def init_backend_options(self, backend_name: str) -> None:
    if backend_name == 'ninja':
        self.options[OptionKey('backend_max_links')] = UserIntegerOption('Maximum number of linker processes to run or 0 for no limit', (0, None, 0))
    elif backend_name.startswith('vs'):
        self.options[OptionKey('backend_startup_project')] = UserStringOption('Default project to execute in Visual Studio', '')