from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar
from .. import logging
def split_state_dict_into_shards_factory(state_dict: Dict[str, TensorT], *, get_tensor_size: TensorSizeFn_T, get_storage_id: StorageIDFn_T=lambda tensor: None, filename_pattern: str=FILENAME_PATTERN, max_shard_size: int=MAX_SHARD_SIZE) -> StateDictSplit:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, Tensor]`):
            The state dictionary to save.
        get_tensor_size (`Callable[[Tensor], int]`):
            A function that returns the size of a tensor in bytes.
        get_storage_id (`Callable[[Tensor], Optional[Any]]`, *optional*):
            A function that returns a unique identifier to a tensor storage. Multiple different tensors can share the
            same underlying storage. This identifier is guaranteed to be unique and constant for this tensor's storage
            during its lifetime. Two tensor storages with non-overlapping lifetimes may have the same id.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.
    """
    storage_id_to_tensors: Dict[Any, List[str]] = {}
    shard_list: List[Dict[str, TensorT]] = []
    current_shard: Dict[str, TensorT] = {}
    current_shard_size = 0
    total_size = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, str):
            logger.info('Skipping tensor %s as it is a string (bnb serialization)', key)
            continue
        storage_id = get_storage_id(tensor)
        if storage_id is not None:
            if storage_id in storage_id_to_tensors:
                storage_id_to_tensors[storage_id].append(key)
                continue
            else:
                storage_id_to_tensors[storage_id] = [key]
        tensor_size = get_tensor_size(tensor)
        if tensor_size > max_shard_size:
            total_size += tensor_size
            shard_list.append({key: tensor})
            continue
        if current_shard_size + tensor_size > max_shard_size:
            shard_list.append(current_shard)
            current_shard = {}
            current_shard_size = 0
        current_shard[key] = tensor
        current_shard_size += tensor_size
        total_size += tensor_size
    if len(current_shard) > 0:
        shard_list.append(current_shard)
    nb_shards = len(shard_list)
    for storage_id, keys in storage_id_to_tensors.items():
        for shard in shard_list:
            if keys[0] in shard:
                for key in keys:
                    shard[key] = state_dict[key]
                break
    if nb_shards == 1:
        filename = filename_pattern.format(suffix='')
        return StateDictSplit(metadata={'total_size': total_size}, filename_to_tensors={filename: list(state_dict.keys())}, tensor_to_filename={key: filename for key in state_dict.keys()})
    tensor_name_to_filename = {}
    filename_to_tensors = {}
    for idx, shard in enumerate(shard_list):
        filename = filename_pattern.format(suffix=f'-{idx + 1:05d}-of-{nb_shards:05d}')
        for key in shard:
            tensor_name_to_filename[key] = filename
        filename_to_tensors[filename] = list(shard.keys())
    return StateDictSplit(metadata={'total_size': total_size}, filename_to_tensors=filename_to_tensors, tensor_to_filename=tensor_name_to_filename)