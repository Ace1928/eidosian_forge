import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@ExperimentalAPI
def get_single_step_input_dict(self, view_requirements: ViewRequirementsDict, index: Union[str, int]='last') -> 'SampleBatch':
    """Creates single ts SampleBatch at given index from `self`.

        For usage as input-dict for model (action or value function) calls.

        Args:
            view_requirements: A view requirements dict from the model for
                which to produce the input_dict.
            index: An integer index value indicating the
                position in the trajectory for which to generate the
                compute_actions input dict. Set to "last" to generate the dict
                at the very end of the trajectory (e.g. for value estimation).
                Note that "last" is different from -1, as "last" will use the
                final NEXT_OBS as observation input.

        Returns:
            The (single-timestep) input dict for ModelV2 calls.
        """
    last_mappings = {SampleBatch.OBS: SampleBatch.NEXT_OBS, SampleBatch.PREV_ACTIONS: SampleBatch.ACTIONS, SampleBatch.PREV_REWARDS: SampleBatch.REWARDS}
    input_dict = {}
    for view_col, view_req in view_requirements.items():
        if view_req.used_for_compute_actions is False:
            continue
        data_col = view_req.data_col or view_col
        if index == 'last':
            data_col = last_mappings.get(data_col, data_col)
            if view_req.shift_from is not None:
                data = self[view_col][-1]
                traj_len = len(self[data_col])
                missing_at_end = traj_len % view_req.batch_repeat_value
                obs_shift = -1 if data_col in [SampleBatch.OBS, SampleBatch.NEXT_OBS] else 0
                from_ = view_req.shift_from + obs_shift
                to_ = view_req.shift_to + obs_shift + 1
                if to_ == 0:
                    to_ = None
                input_dict[view_col] = np.array([np.concatenate([data, self[data_col][-missing_at_end:]])[from_:to_]])
            else:
                input_dict[view_col] = tree.map_structure(lambda v: v[-1:], self[data_col])
        else:
            input_dict[view_col] = self[data_col][index:index + 1 if index != -1 else None]
    return SampleBatch(input_dict, seq_lens=np.array([1], dtype=np.int32))