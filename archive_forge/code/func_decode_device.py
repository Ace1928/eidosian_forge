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
def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != 'cpu' and device.index is None:
        device_interface = get_interface_for_device(device.type)
        return torch.device(device.type, index=device_interface.Worker.current_device())
    return device