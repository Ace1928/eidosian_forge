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
def nn_module_get_all_hooks(mod, check_forward_hooks=False, check_backward_hooks=False, check_state_dict_hooks=False):
    reset_code = torch._C._dynamo.eval_frame.reset_code
    '\n    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre\n    hooks executed during module.__call__, and state_dict hooks which are executed separately.\n    '
    hook_dicts_to_check = []
    check_all_hooks = not check_forward_hooks and (not check_backward_hooks) and (not check_state_dict_hooks)
    if check_forward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(forward_hook_names)
    if check_backward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(backward_hook_names)
    if check_state_dict_hooks:
        hook_dicts_to_check.extend(state_dict_hook_names)
    all_hooks = []
    for hook_dict_name in hook_dicts_to_check:
        hooks = getattr(mod, hook_dict_name, [])
        for hook_name in hooks:
            hook = hooks[hook_name]
            all_hooks.append(hook)
    return all_hooks