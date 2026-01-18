import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import (
from torch.distributed.fsdp import (
from torch.distributed.fsdp._common_utils import (
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
def set_model_state_dict(model: nn.Module, model_state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]], *, options: Optional[StateDictOptions]=None) -> _IncompatibleKeys:
    """Load the model state_dict.

    The counterpart of ``get_model_state_dict`` to set the state_dict to the
    model. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(model, model_state_dict)
    with gc_context():
        info = _verify_options(model, tuple(), optim_only=False, options=options)
        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)