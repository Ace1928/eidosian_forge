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
def gen_gm_and_inputs(target, args, kwargs):
    g = torch.fx.Graph()
    g_args = []
    a_args = []
    for n, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            g_args.append(g.placeholder(f'arg{n}'))
            a_args.append(arg)
        else:
            g_args.append(arg)
    assert all((not isinstance(x, torch.Tensor) for x in kwargs.values()))
    node = g.call_function(target, tuple(g_args), kwargs)
    if len(target._schema.returns) == 1 and str(target._schema.returns[0].type) == 'Tensor':
        node = (node,)
    g.output(node)
    gm = torch.fx.GraphModule({}, g)
    return (gm, a_args)