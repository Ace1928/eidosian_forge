import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.checkpoint._nested_dict import (
from torch.distributed.checkpoint._sharded_tensor_utils import (
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
    if self.flatten_state_dict:
        set_element(self.original_state_dict, self.mappings[read_item.dest_index.fqn], torch.load(value))
    else:
        self.state_dict[read_item.dest_index.fqn] = torch.load(value)