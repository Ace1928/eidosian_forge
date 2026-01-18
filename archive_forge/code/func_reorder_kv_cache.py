import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def reorder_kv_cache(kv_cache: Optional[Tuple], ancestors: torch.Tensor) -> Optional[Tuple]:
    """Re-order the KV-cache based on the ancestors.

    In transformers, the object that stores the KV-cache is a tuple who elements
    are the key cache and the value cache. Each of these caches are tuples where
    each element correpond to a layer. To each layer corresponds a tensor whose
    first dimension is the batch size.

    """
    if kv_cache is None:
        return None
    new_kv_cache: Tuple = tuple()
    for cache_item in kv_cache:
        new_cache_item: Tuple = tuple()
        for layer in cache_item:
            layer = torch.index_select(layer, 0, ancestors.to(layer.device))
            new_cache_item += (layer,)
        new_kv_cache += (new_cache_item,)
    return new_kv_cache