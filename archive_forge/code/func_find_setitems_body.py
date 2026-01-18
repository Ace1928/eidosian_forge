import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def find_setitems_body(setitems, itemsset, loop_body, typemap):
    """
      Find the arrays that are written into (goes into setitems) and the
      mutable objects (mostly arrays) that are written into other arrays
      (goes into itemsset).
    """
    for label, block in loop_body.items():
        find_setitems_block(setitems, itemsset, block, typemap)