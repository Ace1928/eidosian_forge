from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings
import torch
from torch import Tensor, nn
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from . import microbatch
from .async_pipeline import AsyncPipeline
from .async_schedule import Invocation, Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .skip.layout import SkipLayout
from .skip.skippable import Skippable
from .types import LazyModule
def check_balance(module: Union[nn.Sequential, List[LazyModule]], balance: List[int]) -> None:
    if len(module) != sum(balance):
        raise ValueError(f'module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})')
    if any((x <= 0 for x in balance)):
        raise ValueError(f'all balance numbers must be positive integer (balance: {balance})')