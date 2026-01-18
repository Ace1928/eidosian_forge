from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
def can_call(self, func):
    if self.allow_any_calls:
        return True
    if func in self.allowed_calls:
        return True
    owner_method = _unbind_method(func)
    if owner_method and owner_method in self.allowed_calls:
        return True