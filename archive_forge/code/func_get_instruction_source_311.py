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
def get_instruction_source_311(code: types.CodeType, inst: dis.Instruction) -> str:
    """
    Python 3.11+ only. Returns lines of source code (from code object `code`)
    corresponding to `inst`'s location data, and underlines relevant code to `inst`.

    Example: CALL on `g`:
    f(g(
      ^^
        h(x)))
        ^^^^^

    We need our own implementation since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
    assert inst.positions is not None
    if inst.positions.lineno is None:
        return ''
    first_line = linecache.getline(code.co_filename, inst.positions.lineno).rstrip()
    if inst.positions.end_lineno is None:
        return first_line
    if inst.positions.col_offset is None or inst.positions.end_col_offset is None:
        return first_line
    start_offset = _fix_offset(first_line, inst.positions.col_offset)
    end_offset = None
    segment = ''
    markers = []
    if inst.positions.end_lineno == inst.positions.lineno:
        end_offset = _fix_offset(first_line, inst.positions.end_col_offset)
        segment = first_line[start_offset:end_offset]
        markers.append(' ' * start_offset + '~' * (end_offset - start_offset))
    else:
        segment = first_line[start_offset:] + '\n'
        markers.append(' ' * start_offset + '~' * (len(first_line) - start_offset))
        last_line = linecache.getline(code.co_filename, inst.positions.end_lineno).rstrip()
        end_offset = _fix_offset(last_line, inst.positions.end_col_offset)
        for lineno in range(inst.positions.lineno + 1, inst.positions.end_lineno):
            line = linecache.getline(code.co_filename, lineno).rstrip()
            segment += line + '\n'
            num_spaces = len(line) - len(line.lstrip())
            markers.append(' ' * num_spaces + '~' * (len(line) - num_spaces))
        segment += last_line[:end_offset]
        num_spaces = len(last_line) - len(last_line.lstrip())
        markers.append(' ' * num_spaces + '~' * (end_offset - num_spaces))
    anchors: Optional[_Anchors] = None
    try:
        anchors = _extract_anchors_from_expr(segment)
    except AssertionError:
        pass
    if anchors is None:
        markers = [marker.replace('~', '^') for marker in markers]
    else:
        mutable_markers: List[List[str]] = [list(marker) for marker in markers]
        if anchors.left_end_lineno == 0:
            anchors.left_end_offset += start_offset
        if anchors.right_start_lineno == 0:
            anchors.right_start_offset += start_offset
        for lineno in range(len(markers)):
            for col in range(len(mutable_markers[lineno])):
                if lineno < anchors.left_end_lineno:
                    continue
                if lineno == anchors.left_end_lineno and col < anchors.left_end_offset:
                    continue
                if lineno == anchors.right_start_lineno and col >= anchors.right_start_offset:
                    continue
                if lineno > anchors.right_start_lineno:
                    continue
                if mutable_markers[lineno][col] == '~':
                    mutable_markers[lineno][col] = '^'
        markers = [''.join(marker) for marker in mutable_markers]
    result = ''
    for i in range(len(markers)):
        result += linecache.getline(code.co_filename, inst.positions.lineno + i).rstrip() + '\n'
        result += markers[i] + '\n'
    return result