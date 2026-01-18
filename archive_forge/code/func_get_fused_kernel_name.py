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
def get_fused_kernel_name(node_schedule, descriptive_names):
    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == 'original_aten':
        sources = [origin.meta['original_aten']._overloadpacket.__name__ for origin in all_origins if origin.op == 'call_function' and 'original_aten' in origin.meta]
        sources = sorted(set(sources))
    elif descriptive_names == 'torch':
        sources = []
        for origin in all_origins:
            if origin.op == 'call_function' and 'source_fn_stack' in origin.meta:
                source_fn = origin.meta['source_fn_stack'][-1]
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1])
                else:
                    sources.append(source_fn[1].__name__)
        sources = sorted(set(sources))
    elif descriptive_names == 'inductor_node':
        sources = [origin.name for origin in all_origins if origin.op == 'call_function']
    else:
        raise NotImplementedError
    sources = sources
    return '_'.join(['fused'] + sources)