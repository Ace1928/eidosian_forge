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
def can_get_item(self, value, item):
    """Allow accessing `__getiitem__` of allow-listed instances unless it was not modified."""
    return _has_original_dunder(value, allowed_types=self.allowed_getitem, allowed_methods=self._getitem_methods, allowed_external=self.allowed_getitem_external, method_name='__getitem__')