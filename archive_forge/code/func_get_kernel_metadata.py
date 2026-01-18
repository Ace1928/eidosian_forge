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
def get_kernel_metadata(node_schedule, wrapper):
    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == 'call_function']
    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)
    for node in inductor_nodes:
        if 'original_aten' in node.meta:
            key = str(node.meta['original_aten']._overloadpacket)
            original_aten_dict[key].append(node.name)
        if 'from_node' in node.meta:
            key = node.meta['from_node'][0][0]
            from_node_dict[key].append(node.name)
    metadata = f'{wrapper.comment} Source Nodes: [{', '.join(sorted(from_node_dict.keys()))}], Original ATen: [{', '.join(sorted(original_aten_dict.keys()))}]'
    detailed_metadata = []
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(f'{wrapper.comment} {original_node} => {', '.join(sorted(nodes))}')
    return (metadata, '\n'.join(detailed_metadata))