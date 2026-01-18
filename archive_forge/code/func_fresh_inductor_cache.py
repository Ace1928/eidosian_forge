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
@contextlib.contextmanager
def fresh_inductor_cache(cache_entries=None):
    """
    Contextmanager that provides a clean tmp cachedir for inductor.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    with tempfile.TemporaryDirectory() as inductor_cache_dir:
        with mock.patch.dict(os.environ, {'TORCHINDUCTOR_CACHE_DIR': inductor_cache_dir}):
            triton_cache_dir = os.path.join(inductor_cache_dir, 'triton')
            with mock.patch.dict(os.environ, {'TRITON_CACHE_DIR': triton_cache_dir}):
                yield
                if isinstance(cache_entries, dict):
                    assert len(cache_entries) == 0, 'expected empty cache_entries dict'
                    if os.path.exists(triton_cache_dir):
                        files = os.listdir(triton_cache_dir)
                        cache_entries.update({f: os.path.getsize(os.path.join(triton_cache_dir, f)) for f in files if '.lock' not in f})