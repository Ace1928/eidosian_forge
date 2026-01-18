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
def getvaluewithlinemap(self) -> tuple[str, list[tuple[int, LineContext]]]:
    buf = StringIO()
    p = 1
    linemap = []
    for line in self._lines:
        if isinstance(line, DeferredLineBase):
            line = line()
            if line is None:
                continue
        elif isinstance(line, LineContext):
            linemap.append((p, line.context))
            continue
        assert isinstance(line, str)
        buf.write(line)
        buf.write('\n')
        p += 1 + line.count('\n')
    return (buf.getvalue(), linemap)