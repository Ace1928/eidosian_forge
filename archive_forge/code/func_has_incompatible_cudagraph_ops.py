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
def has_incompatible_cudagraph_ops(gm):
    forbidden_set = {'aten._fused_moving_avg_obs_fq_helper.default', 'aten._fused_moving_avg_obs_fq_helper_functional.default', 'aten.multinomial.default', 'fbgemm.dense_to_jagged.default', 'fbgemm.jagged_to_padded_dense.default', 'run_and_save_rng_state', 'run_with_rng_state', 'aten._local_scalar_dense'}
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update({'aten._unsafe_index_put.default', 'aten.index_put.default', 'aten.index_put_.default', 'aten.scatter.src', 'aten.scatter.reduce', 'aten.scatter.value_reduce', 'aten.scatter_add_', 'aten.scatter_add.default', 'aten.scatter_reduce.two', 'aten.scatter_reduce_.two', 'aten.scatter_reduce.two_out'})
    for node in gm.graph.nodes:
        if str(node.target) in forbidden_set:
            return True
    return False