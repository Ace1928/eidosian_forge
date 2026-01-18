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
def add_lang_args(self, lang: str, comp: T.Type['Compiler'], for_machine: MachineChoice, env: 'Environment') -> None:
    """Add global language arguments that are needed before compiler/linker detection."""
    from .compilers import compilers
    self.options.update(compilers.get_global_options(lang, comp, for_machine, env))