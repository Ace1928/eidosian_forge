import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def find_next_users(split_node: torch.fx.Node) -> List[torch.fx.Node]:
    next_users = []
    for getitem_node in split_node.users.keys():
        for getitem_user in getitem_node.users.keys():
            if getitem_user not in next_users:
                next_users.append(getitem_user)
    return next_users