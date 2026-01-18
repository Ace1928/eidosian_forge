import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple
from ._base import FILENAME_PATTERN, MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory
def get_storage_size(tensor: 'torch.Tensor') -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L31C1-L41C59
    """
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        try:
            return tensor.storage().size() * _get_dtype_size(tensor.dtype)
        except NotImplementedError:
            return tensor.nelement() * _get_dtype_size(tensor.dtype)