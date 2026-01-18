import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def append_stacktrace_summary(node: Node):
    """
            Append a summary of the stacktrace to the generated code. This is
            useful for debugging.
            """
    nonlocal prev_stacktrace
    if node.op not in {'placeholder', 'output'}:
        if node.stack_trace:
            if node.stack_trace != prev_stacktrace:
                prev_stacktrace = node.stack_trace
                summary_str = ''
                parsed_stack_trace = _parse_stack_trace(node.stack_trace)
                if parsed_stack_trace is not None:
                    lineno = parsed_stack_trace.lineno
                    code = parsed_stack_trace.code
                    summary_str = f'File: {parsed_stack_trace.file}:{lineno}, code: {code}'
                body.append(f'\n# {summary_str}\n')
        elif prev_stacktrace != '':
            prev_stacktrace = ''
            body.append('\n# No stacktrace found for following nodes\n')