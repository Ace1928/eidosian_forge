import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List
import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, RandomValueSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable
def odict_getitem(self, tx, key):
    from .builder import VariableBuilder
    index = key.source if ConstDictVariable.is_valid_key(key) and key.source is not None else key.as_python_constant()
    return VariableBuilder(tx, ODictGetItemSource(self.source, index))(collections.OrderedDict.__getitem__(self.value, key.as_python_constant()))