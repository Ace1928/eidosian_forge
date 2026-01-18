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
def parse_cmd_line_options(args: SharedCMDOptions) -> None:
    args.cmd_line_options = create_options_dict(args.projectoptions)
    for key in chain(BUILTIN_OPTIONS.keys(), (k.as_build() for k in BUILTIN_OPTIONS_PER_MACHINE.keys()), BUILTIN_OPTIONS_PER_MACHINE.keys()):
        name = str(key)
        value = getattr(args, name, None)
        if value is not None:
            if key in args.cmd_line_options:
                cmdline_name = BuiltinOption.argparse_name_to_arg(name)
                raise MesonException(f'Got argument {name} as both -D{name} and {cmdline_name}. Pick one.')
            args.cmd_line_options[key] = value
            delattr(args, name)