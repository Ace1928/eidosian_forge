import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from enum import auto, Enum
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParameter
from ._optim_utils import (
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
from .wrap import CustomPolicy, ModuleWrapPolicy
@staticmethod
def rekey_optim_state_dict(optim_state_dict: Dict[str, Any], optim_state_key_type: OptimStateKeyType, model: torch.nn.Module, optim_input: Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]=None, optim: Optional[torch.optim.Optimizer]=None) -> Dict[str, Any]:
    """Re-keys the optimizer state dict ``optim_state_dict`` to use the key type ``optim_state_key_type``.

        This can be used to achieve compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> osd = nonwrapped_optim.state_dict()
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
            >>> wrapped_model, wrapped_optim = ...
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
            >>> wrapped_optim.load_state_dict(sharded_osd)

        Returns:
            Dict[str, Any]: The optimizer state dict re-keyed using the
            parameter keys specified by ``optim_state_key_type``.
        """
    FullyShardedDataParallel._warn_optim_input(optim_input)
    using_optim_input = FullyShardedDataParallel._is_using_optim_input(optim_input, optim)
    assert optim_state_key_type in (OptimStateKeyType.PARAM_NAME, OptimStateKeyType.PARAM_ID)
    osd = optim_state_dict
    uses_param_name_mask = [type(param_key) is str for param_key in osd['state']]
    uses_param_id_mask = [type(param_key) is int for param_key in osd['state']]
    if any(uses_param_name_mask) and (not all(uses_param_name_mask)) or (any(uses_param_id_mask) and (not all(uses_param_id_mask))):
        error_msg = f'Invalid parameter keys: {osd['state'].keys()}'
        raise ValueError(error_msg)
    if optim_state_key_type == OptimStateKeyType.PARAM_NAME and all(uses_param_name_mask) or (optim_state_key_type == OptimStateKeyType.PARAM_ID and all(uses_param_id_mask)):
        return osd
    new_osd = {}
    if optim_state_key_type == OptimStateKeyType.PARAM_NAME:
        param_id_to_param = _get_param_id_to_param_from_optim_input(model, optim_input) if using_optim_input else _get_param_key_to_param(optim)
        param_to_param_name = _get_param_to_fqn(model)
        param_id_to_param_name: List[str] = [param_to_param_name[param] for param in param_id_to_param.values()]
        new_osd['state'] = {param_id_to_param_name[param_id]: param_state for param_id, param_state in osd['state'].items()}
        new_osd['param_groups'] = copy.deepcopy(osd['param_groups'])
        for param_group in new_osd['param_groups']:
            param_group['params'] = sorted([param_id_to_param_name[param_id] for param_id in param_group['params']])
        return new_osd
    elif optim_state_key_type == OptimStateKeyType.PARAM_ID:
        param_name_to_param = _get_fqn_to_param(model)
        param_to_param_id = _get_param_to_param_id_from_optim_input(model, optim_input) if using_optim_input else _get_param_to_param_key(optim)
        param_name_to_param_id = {param_name: param_to_param_id[param] for param_name, param in param_name_to_param.items() if param in param_to_param_id}
        new_osd['state'] = {param_name_to_param_id[param_name]: param_state for param_name, param_state in osd['state'].items()}
        new_osd['param_groups'] = copy.deepcopy(osd['param_groups'])
        for param_group in new_osd['param_groups']:
            param_group['params'] = sorted([param_name_to_param_id[param_name] for param_name in param_group['params']])
        return new_osd
    return new_osd