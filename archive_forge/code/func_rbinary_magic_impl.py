import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def rbinary_magic_impl(self, other):
    self = promote(self)
    other = promote(other)
    if is_constant(self):
        return method_to_operator(method)(get_constant(self), other)
    if is_constant(other):
        other = get_constant(other)
    other_node = to_node(self.node, other)
    if other_node is NotImplemented:
        return NotImplemented
    ret = wrap_node(getattr(other_node, method_attr)(self.node))
    return get_constant(ret) if is_constant(ret) else ret