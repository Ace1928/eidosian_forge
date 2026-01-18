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
def emit_base_options_warnings(self, enabled_opts: T.List[OptionKey]) -> None:
    if OptionKey('b_bitcode') in enabled_opts:
        mlog.warning("Base option 'b_bitcode' is enabled, which is incompatible with many linker options. Incompatible options such as 'b_asneeded' have been disabled.", fatal=False)
        mlog.warning('Please see https://mesonbuild.com/Builtin-options.html#Notes_about_Apple_Bitcode_support for more details.', fatal=False)