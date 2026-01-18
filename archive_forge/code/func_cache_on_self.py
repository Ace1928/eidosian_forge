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
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    key = f'__{fn.__name__}_cache'

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)

    def clear_cache(self):
        if hasattr(self, key):
            delattr(self, key)
    wrapper.clear_cache = clear_cache
    return wrapper