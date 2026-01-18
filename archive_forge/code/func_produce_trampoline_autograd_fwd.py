import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
def produce_trampoline_autograd_fwd(fn_cls):

    def trampoline_autograd_fwd(*args, **kwargs):
        return fn_cls.forward(*args, **kwargs)
    trampoline_autograd_fwd._origin = produce_trampoline_autograd_fwd
    return trampoline_autograd_fwd