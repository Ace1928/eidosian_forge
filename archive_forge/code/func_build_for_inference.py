import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def build_for_inference(self) -> SampleBatch:
    """During inference, we will build a SampleBatch with a batch size of 1 that
        can then be used to run the forward pass of a policy. This data will only
        include the enviornment context for running the policy at the last timestep.

        Returns:
            A SampleBatch with a batch size of 1.
        """
    batch_data = {}
    np_data = {}
    for view_col, view_req in self.view_requirements.items():
        data_col = view_req.data_col or view_col
        if not view_req.used_for_compute_actions:
            continue
        if np.any(view_req.shift_arr > 0):
            raise ValueError(f'During inference the agent can only use past observations to respect causality. However, view_col = {view_col} seems to depend on future indices {view_req.shift_arr}, while the used_for_compute_actions flag is set to True. Please fix the discrepancy. Hint: If you are using a custom model make sure the view_requirements are initialized properly and is point only refering to past timesteps during inference.')
        if data_col not in self.buffers:
            self._fill_buffer_with_initial_values(data_col, view_req, build_for_inference=True)
            self._prepare_for_data_cols_with_dummy_values(data_col)
        self._cache_in_np(np_data, data_col)
        data = []
        for d in np_data[data_col]:
            element_at_t = d[view_req.shift_arr + len(d) - 1]
            if element_at_t.shape[0] == 1:
                data.append(element_at_t)
                continue
            data.append(element_at_t[None])
        batch_data[view_col] = self._unflatten_as_buffer_struct(data, data_col)
    batch = self._get_sample_batch(batch_data)
    return batch