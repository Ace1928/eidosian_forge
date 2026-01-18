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
def read_cmd_line_file(build_dir: str, options: SharedCMDOptions) -> None:
    filename = get_cmd_line_file(build_dir)
    if not os.path.isfile(filename):
        return
    config = CmdLineFileParser()
    config.read(filename)
    d = {OptionKey.from_string(k): v for k, v in config['options'].items()}
    d.update(options.cmd_line_options)
    options.cmd_line_options = d
    properties = config['properties']
    if not options.cross_file:
        options.cross_file = ast.literal_eval(properties.get('cross_file', '[]'))
    if not options.native_file:
        options.native_file = ast.literal_eval(properties.get('native_file', '[]'))