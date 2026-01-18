import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
def program_ids_from_grid(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    reversed_grid = reversed(grid)
    ranges_for_each_dimension = [range(dim) for dim in reversed_grid]
    index_combinations = list(itertools.product(*ranges_for_each_dimension))
    random.shuffle(index_combinations)
    for index_combination in index_combinations:
        yield index_combination