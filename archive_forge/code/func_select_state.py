import gymnasium as gym
from typing import Optional, List, Dict
from ray.rllib.algorithms.sac.sac_torch_model import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def select_state(self, state_batch: List[TensorType], net: List[str]) -> Dict[str, List[TensorType]]:
    assert all((n in ['policy', 'q', 'twin_q'] for n in net)), 'Selected state must be either for policy, q or twin_q network'
    policy_state_len = len(self.action_model.get_initial_state())
    q_state_len = len(self.q_net.get_initial_state())
    selected_state = {}
    for n in net:
        if n == 'policy':
            selected_state[n] = state_batch[:policy_state_len]
        elif n == 'q':
            selected_state[n] = state_batch[policy_state_len:policy_state_len + q_state_len]
        elif n == 'twin_q':
            if self.twin_q_net:
                selected_state[n] = state_batch[policy_state_len + q_state_len:]
            else:
                selected_state[n] = []
    return selected_state