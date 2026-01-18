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
def aggregate_origins(node_schedule):
    from . import ir
    if isinstance(node_schedule, list):
        return functools.reduce(operator.or_, [node.node.origins for node in node_schedule if hasattr(node, 'node') and node.node], set())
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return set()