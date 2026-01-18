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
def is_pointwise_use(use):
    if not use.op == 'call_function':
        return False
    if not (isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem):
        return False
    if use.target is operator.getitem or is_view(use.target):
        return all((is_pointwise_use(u) for u in use.users))
    return torch.Tag.pointwise in use.target.tags