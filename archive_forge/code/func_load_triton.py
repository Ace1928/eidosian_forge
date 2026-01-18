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
@functools.lru_cache(None)
def load_triton():
    try:
        from triton.testing import do_bench as triton_do_bench
    except ImportError as exc:
        raise NotImplementedError('requires Triton') from exc
    return (triton_do_bench, 'quantiles' if inspect.signature(triton_do_bench).parameters.get('quantiles') is not None else 'percentiles')