import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
def register_backend_for_device(device: str, device_scheduling: type, device_wrapper_codegen: type):
    device_codegens[device] = DeviceCodegen(device_scheduling, device_wrapper_codegen)