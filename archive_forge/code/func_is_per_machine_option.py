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
@staticmethod
def is_per_machine_option(optname: OptionKey) -> bool:
    if optname.name in BUILTIN_OPTIONS_PER_MACHINE:
        return True
    return optname.lang is not None