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
def copy_build_options_from_regular_ones(self) -> bool:
    dirty = False
    assert not self.is_cross_build()
    for k in BUILTIN_OPTIONS_PER_MACHINE:
        o = self.options[k]
        dirty |= self.options[k.as_build()].set_value(o.value)
    for bk, bv in self.options.items():
        if bk.machine is MachineChoice.BUILD:
            hk = bk.as_host()
            try:
                hv = self.options[hk]
                dirty |= bv.set_value(hv.value)
            except KeyError:
                continue
    return dirty