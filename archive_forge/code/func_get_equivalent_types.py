import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node
import torch
from torch.fx.passes.utils.source_matcher_utils import (
def get_equivalent_types() -> List[Set]:
    return _EQUIVALENT_TYPES