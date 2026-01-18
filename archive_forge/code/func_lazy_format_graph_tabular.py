import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
def lazy_format_graph_tabular(fn_name, gm):

    def inner():
        try:
            from tabulate import tabulate
        except ImportError:
            return 'Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n' + str(lazy_format_graph_code(fn_name, gm))
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes]
        graph_str = tabulate(node_specs, headers=['opcode', 'name', 'target', 'args', 'kwargs'])
        return _format_graph_code(fn_name, gm.forward.__code__.co_filename, graph_str)
    return LazyString(inner)