import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def construct_rpc_backend_options(backend, rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC, init_method=rpc_constants.DEFAULT_INIT_METHOD, **kwargs):
    return backend.value.construct_rpc_backend_options_handler(rpc_timeout, init_method, **kwargs)