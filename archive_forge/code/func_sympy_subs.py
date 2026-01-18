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
def sympy_subs(expr: sympy.Expr, replacements: Dict[Any, Any]) -> sympy.Expr:
    """
    xreplace is faster than subs, but is way more picky
    """

    def promote_strings(key):
        if isinstance(key, str):
            return sympy_symbol(key)
        return key
    return sympy.sympify(expr).xreplace({promote_strings(k): promote_strings(v) for k, v in replacements.items()})