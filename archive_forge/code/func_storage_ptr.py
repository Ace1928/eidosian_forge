import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple
from ._base import FILENAME_PATTERN, MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory
def storage_ptr(tensor: 'torch.Tensor') -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L11C1-L20C21.
    """
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            return 0