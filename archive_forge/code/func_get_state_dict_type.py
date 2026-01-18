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
def get_state_dict_type(module: nn.Module) -> StateDictSettings:
    """Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at ``module``.

        The target module does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
    state_dict_settings: Optional[StateDictSettings] = None
    for submodule in FullyShardedDataParallel.fsdp_modules(module):
        if state_dict_settings is None:
            state_dict_settings = StateDictSettings(state_dict_type=submodule._state_dict_type, state_dict_config=submodule._state_dict_config, optim_state_dict_config=submodule._optim_state_dict_config)
            _set_optim_use_dtensor(submodule, state_dict_settings)
        else:
            submodule_settings = StateDictSettings(submodule._state_dict_type, submodule._state_dict_config, submodule._optim_state_dict_config)
            assert state_dict_settings == submodule_settings, f'All FSDP modules must have the same state dict settings.Got {submodule_settings} and {state_dict_settings}.'
            _set_optim_use_dtensor(submodule, submodule_settings)
    return state_dict_settings