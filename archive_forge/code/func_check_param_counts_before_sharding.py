import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
def check_param_counts_before_sharding(full_optim_state_dict: Dict, n_instances: int) -> None:
    n_local_params_in_opt = len(set(full_optim_state_dict['param_id_map'].values()))
    msg = f'Including itself, this model has {n_instances} nested instances. When the optimizer state was saved there were {n_local_params_in_opt}'
    stateless = len(full_optim_state_dict['state']) == 0
    assert stateless or n_instances == n_local_params_in_opt, msg