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
def create_default_global_save_plan(all_plans: List[SavePlan], rewrite_index_hints: bool=True) -> Tuple[List[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.
    """
    md: Dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md
            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(TensorStorageMetadata, md.setdefault(item.index.fqn, TensorStorageMetadata(properties=item.tensor_data.properties, size=item.tensor_data.size, chunks=[])))
                new_item = item
                if rewrite_index_hints:
                    new_index = dataclasses.replace(item.index, index=len(tensor_md.chunks))
                    new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)
                assert item.tensor_data.chunk is not None, f'\n                    Cannot create MD for tensor without bounds.\n                    FQN: {item.index.fqn}\n                '
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))