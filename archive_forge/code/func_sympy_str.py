from __future__ import annotations
import collections
import contextlib
import enum
import functools
import getpass
import inspect
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
from unittest import mock
import sympy
from typing_extensions import Concatenate, ParamSpec
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """
    if isinstance(expr, sympy.Symbol):
        return expr.name
    if isinstance(expr, sympy.Add):
        return ' + '.join(map(sympy_str, expr.args))
    if isinstance(expr, sympy.Mul):
        return ' * '.join(map(sympy_str, expr.args))
    if isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv)):
        return f'{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})'
    return str(expr)