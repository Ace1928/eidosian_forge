import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
def get_parameter_type(line):
    item_type = line[line.find(type_comment) + len(type_comment):]
    return item_type.strip()