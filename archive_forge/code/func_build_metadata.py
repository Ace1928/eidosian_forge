from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata:
    check_tensor(self.shards, tensor_sizes)
    return sharded_tensor_meta.ShardedTensorMetadata(self.shards, tensor_sizes, tensor_properties)