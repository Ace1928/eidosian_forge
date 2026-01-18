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
def is_cross_build(self, when_building_for: MachineChoice=MachineChoice.HOST) -> bool:
    if when_building_for == MachineChoice.BUILD:
        return False
    return len(self.cross_files) > 0