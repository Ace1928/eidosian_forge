from torch.fx import (
from torch.fx.graph import Graph
from .match_utils import (
from .pattern_utils import (
from ..backend_config import (
from ..backend_config.utils import (
from .custom_config import FuseCustomConfig
from .fuse_handler import (
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings
from torch.ao.quantization.utils import Pattern, NodePattern
def default_root_node_getter(node_pattern):
    while not isinstance(node_pattern[-1], Node):
        node_pattern = node_pattern[-1]
    return node_pattern[-1]