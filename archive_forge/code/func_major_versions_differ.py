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
def major_versions_differ(v1: str, v2: str) -> bool:
    v1_major, v1_minor = v1.rsplit('.', 1)
    v2_major, v2_minor = v2.rsplit('.', 1)
    return v1_major != v2_major or ('99' in {v1_minor, v2_minor} and v1_minor != v2_minor)