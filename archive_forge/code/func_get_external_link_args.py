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
def get_external_link_args(self, for_machine: MachineChoice, lang: str) -> T.List[str]:
    return T.cast('T.List[str]', self.options[OptionKey('link_args', machine=for_machine, lang=lang)].value)