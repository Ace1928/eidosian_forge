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
def guard_int(self, file, line):
    r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
    try:
        return int(r)
    except Exception:
        log.warning('Failed to convert to int: %s', r)
        raise