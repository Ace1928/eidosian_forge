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
def can_operate(self, dunders: Tuple[str, ...], a, b=None):
    objects = [a]
    if b is not None:
        objects.append(b)
    return all([_has_original_dunder(obj, allowed_types=self.allowed_operations, allowed_methods=self._operator_dunder_methods(dunder), allowed_external=self.allowed_operations_external, method_name=dunder) for dunder in dunders for obj in objects])