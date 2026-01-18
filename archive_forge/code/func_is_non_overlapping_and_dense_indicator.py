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
def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> 'SymNode':
    return self._is_non_overlapping_and_dense_indicator(sizes, strides)